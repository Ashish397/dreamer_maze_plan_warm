import copy
import torch

import networks
import tools

import numpy as np
import torch.nn.functional as F

from torch import nn
to_np = lambda x: x.detach().cpu().numpy()

class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        # shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.m_encoder = networks.MultiEncoder(shapes, **config.m_encoder)
        self.v_encoder = networks.MultiEncoder(shapes, **config.v_encoder)
        self.embed_size = self.v_encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["entropy"] = networks.MLP(
            feat_size,
            (255,) if config.entropy_head["dist"] == "symlog_disc" else (),
            config.entropy_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.entropy_head["dist"],
            outscale=config.entropy_head["outscale"],
            device=config.device,
            name="Entropy_hat",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            entropy_hat=config.entropy_head["loss_scale"],
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.amp.autocast('cuda'):
                embed = self.v_encoder(data) + self.m_encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                posterior_entropy = self.dynamics.get_dist(post).entropy()
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    if name == 'entropy':
                        feat = self.dynamics.get_feat(prior)                    
                    else:
                        feat = self.dynamics.get_feat(post)                    
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    if name in ['image', 'map']:
                        loss = -pred.log_prob(data[name], name)
                    elif name == "entropy":
                        # Entropy loss: Mean squared error to match the posterior entropy
                        loss = -pred.log_prob(posterior_entropy)
                    else:
                        loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.amp.autocast('cuda'):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = {
            k: torch.tensor(v, device='cuda', dtype=torch.float32)
            for k, v in obs.items()
        }
        # obs["image"] = obs["image"] / 255.0
        # obs["map"] = obs["map"] / 255.0
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = obs["discount"].unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = (1.0 - obs["is_terminal"])
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.v_encoder(data) + self.m_encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon_v = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        recon_m = self.heads["decoder"](self.dynamics.get_feat(states))["map"].mode()[
            :6
        ]
        recon = torch.cat([recon_v, recon_m], dim=-2)
        # reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl_v = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        openl_m = self.heads["decoder"](self.dynamics.get_feat(prior))["map"].mode()
        openl = torch.cat([openl_v, openl_m], dim=-2)
        # reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = torch.cat([data["image"], data["map"]], dim=-2)[:6]
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)

    def video_pred_n_step(self, data, start_steps = 5, lookahead = 5):
        data = self.preprocess(data)
        #ground truth output pregenerated
        truth = torch.cat([data["image"], data["map"]], dim=-2)
        #model output preprepared for memory reasons
        model = torch.zeros_like(truth)

        #Generate embedding and states
        embed = self.v_encoder(data) + self.m_encoder(data)
        states, _ = self.dynamics.observe(
            embed[:, :start_steps], data["action"][:, :start_steps], data["is_first"][:, :start_steps]
        )

        #Generate reconstruction
        recon_v = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()
        recon_m = self.heads["decoder"](self.dynamics.get_feat(states))["map"].mode()
        recon = torch.cat([recon_v, recon_m], dim=-2)
        
        #Infill with the reconstructed starting steps
        model[:,:start_steps] = recon

        ##Don't know what to do with reward lines right now so muting them 
        ##      - note that the second line should be lower down after prior is generated
        # reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        # reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()

        #Let's start predicting
        num_predict_steps = np.ceil((truth.shape[1] - start_steps) / lookahead).astype(int)

        for predict_step in range(num_predict_steps):
            if predict_step > 0:
                states, _ = self.dynamics.observe(
                    embed[:, :start_steps + predict_step * lookahead], data["action"][:, :start_steps + predict_step * lookahead], data["is_first"][:, :start_steps + predict_step * lookahead]
                )
            init = {k: v[:, -1] for k, v in states.items()}
            prior = self.dynamics.imagine_with_action(data["action"][:, start_steps + predict_step * lookahead : start_steps + (predict_step + 1) * lookahead], init)
            openl_v = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
            openl_m = self.heads["decoder"](self.dynamics.get_feat(prior))["map"].mode()
            openl = torch.cat([openl_v, openl_m], dim=-2)
            model[:, start_steps + predict_step * lookahead : start_steps + (predict_step + 1) * lookahead] = openl

        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def video_reconstruct(self, data):
        data = self.preprocess(data)
        #ground truth output pregenerated
        truth = torch.cat([data["image"], data["map"]], dim=-2)

        #Generate embedding and states
        embed = self.v_encoder(data) + self.m_encoder(data)
        states, _ = self.dynamics.observe(
            embed, data["action"], data["is_first"]
        )

        #Generate reconstruction
        recon_v = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()
        recon_m = self.heads["decoder"](self.dynamics.get_feat(states))["map"].mode()
        recon = torch.cat([recon_v, recon_m], dim=-2)
        
        ##Don't know what to do with reward lines right now so muting them 
        ##      - note that the second line should be lower down after prior is generated
        # reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        # reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()

        error = (recon - truth + 1.0) / 2.0
        return torch.cat([truth, recon, error], 2)

class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        """
        Initialize the ImagBehavior module.

        Args:
            config: Configuration object containing hyperparameters.
            world_model: The world model used for imagining trajectories.
        """
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer(
                "ema_vals", torch.zeros((2,), device=self._config.device)
            )
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        """
        Perform one training step for the actor and value networks.

        Args:
            start: Initial states to start imagining trajectories from.
            objective: Function to compute rewards from imagined features, states, and actions.

        Returns:
            Tuple containing imagined features, states, actions, weights, and metrics.
        """
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.amp.autocast('cuda'):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.amp.autocast('cuda'):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        """
        Imagine trajectories starting from the given initial states.

        Args:
            start: Initial states to start imagining from.
            policy: Policy to generate actions.
            horizon: Number of steps to imagine.
            spit_stats (bool): If True, also return statistics from dynamics.

        Returns:
            If spit_stats is False:
                feats: Imagined features.
                states: Imagined states.
                actions: Imagined actions.
            If spit_stats is True:
                feats: Imagined features.
                states: Imagined states.
                actions: Imagined actions.
                stats: Dynamics statistics.
        """
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions
        
    def _compute_target(self, imag_feat, imag_state, reward):
        """
        Compute the target values and weights for value network training.

        Args:
            imag_feat: Imagined features.
            imag_state: Imagined states.
            reward: Imagined rewards.

        Returns:
            target: Computed target values.
            weights: Discounted weights for each time step.
            base: Baseline value predictions.
        """
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        """
        Compute the actor loss for policy optimization.

        Args:
            imag_feat: Imagined features.
            imag_action: Imagined actions.
            target: Target values computed from rewards.
            weights: Discounted weights for each time step.
            base: Baseline value predictions.

        Returns:
            actor_loss: Computed loss for the actor network.
            metrics: Dictionary of metrics for logging.
        """
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        """
        Update the slow-moving target network for the value function.

        Updates the parameters of the slow target network by interpolating with the current value network parameters.
        """
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

    def _ent(self, p):
        p = F.softmax(p, dim=-1)
        #could just mean the whole thing but just in case
        return - torch.mean(torch.sum(p * torch.log(p+1e-3), dim=-1), dim=-1)

    def _kl_loss(self, post, prior):
        #insignificant change - changed from torch.softmax to f.softmax if that breaks later code runs :)
        return F.kl_div(F.softmax(post, dim=-1), F.softmax(prior, dim=-1))
        