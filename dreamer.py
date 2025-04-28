import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools

import torch
from torch import nn

import os
import pathlib
import random
import torch.cuda.amp as amp

from tensordict import TensorDict

class Dreamer(nn.Module):
    def __init__(self, obs_space, config, logger):
        super(Dreamer, self).__init__()
        #Underlying - don't change
        self._step = logger.step // config.action_repeat
        self._config = config
        self._train_steps = config.train_steps
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._wm = models.WorldModel(obs_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, None),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)
        self._update_count = 0

        #Plan parameters
        self._plan_choices = config.plan_choices
        self._planned_traj = None

        #Plan Variables
        self._replan_traj = False
        self._plan_pos = 0
        self._in_plan = False

        self._plan_bin = False
        self._cancel_bin = False
        self._ent_weight = None
        self._plan_hor = None
        self._plan_mask = [1,1,1]
        self._final_state = None

        #Plan Metrics
        self._traj_ents = None
        self._curcum_diff = None
        self._metrics = {}
        self._update_count_plan = 0

        self._plan_b = expl.PlanBehavior(
            config=self._config,
            world_model=self._wm,
            taskbehavior=self._task_behavior
        )

        self._planner_train_step = 0

    def __call__(self, obs, reset, this_batch, given_step, state=None, training=True, data_train=True):
        step = self._step
        self._planner_train_step += 1
        if data_train:
            for _ in range(self._train_steps):
                self._train(this_batch)
                self._update_count += 1
                self._metrics["update_count"] = self._update_count #count the number of times trained
        if self._planner_train_step % self._plan_b.train_every == 0:
            self._plan_b._train(logger=self._logger) #train the planner and count it
            self._update_count_plan += 1
            self._metrics["update_count_plan"] = self._update_count_plan #count the number of times trained
        if self._should_log(step):
            for name, values in self._metrics.items():
                if isinstance(values, int) or isinstance(values, float):
                    # Make it into a list on the fly
                    values = [values]
                # Now values is a list, so we can safely filter out None
                numeric_values = [v for v in values if v is not None]

                if numeric_values:
                    self._logger.scalar(name, float(np.mean(numeric_values)))
                    if name in ['len_b4_replan', '_plan_prob_metric', '_plan_canc_prob_metric', '_plan_hor_metric', '_plan_ent_metric']:
                        self._logger.scalar(name+'_std', float(np.std(numeric_values)))    
                        self._logger.scalar(name+'_max', float(np.max(numeric_values)))    
                else:
                    pass  # or handle empty lists differently

                self._metrics[name] = []
                
        #What would the plan params be?
        meta_sec_actions, ap_logprob_sec, start_state = self.meta_policy_sec(obs, state, given_step)
        
        #Should we enact a new plan?
        action_plan, ap_logprob_prim, start_state = self.meta_policy_prim(start_state, meta_sec_actions)
        this_prob = action_plan[0].detach().cpu().item() / 4

        self._metrics.setdefault("_plan_prob_metric", []).append(this_prob)

        #:-1539
        if random.random() > (this_prob * this_prob):
            #No need to save the secondary metrics if a plan wasn't enacted
            meta_sec_actions = None
            ap_logprob_sec = None
            implemented = False
        else:
            #We go into plan
            self._in_plan = True
            #A plan will be enacted
            implemented = True
            #A new traj will be planned
            self._replan_traj = True
            #Get the actions out of the object
            self._plan_hor, self._ent_weight = meta_sec_actions[0]
            #Convert them into real parameters
            self._plan_hor = int(((self._plan_hor + 1) / 5) * 15)
            self._ent_weight = float(self._ent_weight / 4)
            #Log them
            self._metrics.setdefault("_plan_hor_metric", []).append(self._plan_hor)
            self._metrics.setdefault("_plan_ent_metric", []).append(self._ent_weight)


        self._plan_b._add_to_buffer(
            observation = start_state,
            action = action_plan,
            sample_log_prob = ap_logprob_prim,
            action_sec = meta_sec_actions,
            sample_log_prob_sec = ap_logprob_sec,
            implemented = implemented,
            mode = 'begin'
        )

        if self._in_plan:
            policy_output, state = self._planned_policy(obs, state)
            self._metrics.setdefault("_plan_steps_metric", []).append(1)
        else:
            policy_output, state = self._policy(obs, state, training)
            self._metrics.setdefault("_plan_steps_metric", []).append(0)

        policy_output['action'] = policy_output['action'].squeeze()
        
        if data_train:
            self._step += 1 if type(reset) == bool else len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def meta_policy_prim(self, start_state, sec_act):
        start_state = torch.cat([
            start_state,
            sec_act,
        ], dim=-1) 
        out_old = self._plan_b.policy_module_prim(TensorDict({"observation": start_state}, batch_size=[1]))
        return out_old["action"], out_old["sample_log_prob"], start_state

    def meta_policy_sec(self, obs, state, given_step):
        #what would the greedy action be?
        greedy_output, _ = self._policy(obs, state, False, dontadd=True)
        greedy_output = greedy_output['action']

        if state is None:
            latent = action = None
        else:
            latent, action = state

        obs = self._wm.preprocess(obs)
        embed = self._wm.v_encoder(obs) + self._wm.m_encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed.unsqueeze(0), obs["is_first"])
                
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent).squeeze()

        start_state = torch.cat([
            embed.unsqueeze(0), 
            feat.unsqueeze(0), 
            torch.tensor([[given_step / 4096.0]], device='cuda'), 
            greedy_output,
            torch.tensor([[self._plan_pos / self._config.plan_max_horizon]], device='cuda'),
            torch.tensor([[self._in_plan]], device='cuda')
            ], dim=-1)

        out_old = self._plan_b.policy_module_sec(TensorDict({"observation": start_state}, batch_size=[1]))

        finfeat = torch.zeros_like(feat.unsqueeze(0)) if self._final_state is None else self._final_state
        
        start_state = torch.cat([
            start_state,
            finfeat,
        ], dim=-1)

        return out_old["action"], out_old["sample_log_prob"].sum(dim=-1), start_state

    def _policy(self, obs, state, training, dontadd=False):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.v_encoder(obs) + self._wm.m_encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed.unsqueeze(0), obs["is_first"])
        if not dontadd:
            self._plan_b._add_to_buffer(mode = 'entropy', 
                                            entropy = self._wm.dynamics.get_dist(latent).entropy())
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state
    
    def _plan_trajectory(self, obs, state):

        # Initialize or update metrics
        if not obs["is_first"]:
            self._metrics.setdefault("actual_ent_diff", []).append(self._curcum_diff)
            if self._plan_pos != 0:
                self._metrics.setdefault("len_b4_replan", []).append(self._plan_pos)

        if type(self._metrics.setdefault("num_replans", 0)) == list:
            self._metrics["num_replans"] = 0
            
        self._metrics["num_replans"] = self._metrics.setdefault("num_replans", 0) + 1

        # Initialize latent and action
        if state is None:
            latent = action = None
        else:
            latent, action = state

        # Preprocess observation
        obs = self._wm.preprocess(obs)
        embed = self._wm.v_encoder(obs) + self._wm.m_encoder(obs)
        latent, _, = self._wm.dynamics.obs_step(latent, action, embed.unsqueeze(0), obs["is_first"])

        # Initialize trajectory
        inp_act, ent, ent_mean, rew_mean, self._final_state = self._plan_b._flow(latent, horizon=self._plan_hor, r_weight=self._ent_weight, num_choices=self._plan_choices)
        ref_act = inp_act.clone().detach().mean(1)

        self._metrics.setdefault("ent_mean", []).append(ent_mean.cpu())
        self._metrics.setdefault("rew_mean", []).append(rew_mean.cpu())
        
        # ent shape is 15, 16, or timesteps, num_choices
        all_ents = ent.sum(0)

        final_reward_diff = all_ents.max() - all_ents.mean()
        self._metrics.setdefault("final_reward_diff", []).append(final_reward_diff)

        inp_act = inp_act[:,torch.argmax(all_ents).item()]

        ent = ent[:,torch.argmax(all_ents).item()]

        # Compute final trajectory difference
        final_delta = (inp_act - ref_act).mean(dim=0).mean().item()
        self._metrics.setdefault("final_delta", []).append(final_delta)

        # Finalize planned trajectory
        self._planned_traj = inp_act.detach()
        self._traj_ents = ent.detach()
        self._plan_pos = 0
        self._curcum_diff = 0
        self._final_state = None
        self._replan_traj = False
        
    def _planned_policy(self, obs, state):
        # Replan trajectory if needed
        if self._replan_traj:
            self._plan_trajectory(obs, state)

        # Initialize latent and action
        latent = None if state is None else state[0]
        action = self._planned_traj[self._plan_pos].unsqueeze(0)

        # Preprocess observation
        obs = self._wm.preprocess(obs)
        embed = self._wm.v_encoder(obs) + self._wm.m_encoder(obs)
        latent, _, _, post_stats = self._wm.dynamics.obs_step_stats(
            latent, action, embed.unsqueeze(0), obs["is_first"], ent_type='dream'
        )
        self._plan_b._add_to_buffer(mode = 'entropy',
                                        entropy = self._wm.dynamics.get_dist(latent).entropy())

        this_ent = post_stats.mean().item()
        self._curcum_diff += abs(this_ent - self._traj_ents[self._plan_pos].item())

        # Prepare policy output
        feat = self._wm.dynamics.get_feat(latent)
        actor = self._task_behavior.actor(feat)
        logprob = actor.log_prob(action)

        # Advance trajectory position
        self._plan_pos += 1
        if self._plan_pos >= len(self._planned_traj):
            self._in_plan = False
            self._plan_pos = 0
            self._final_state = None

        # Return policy output and updated state
        latent = {k: v.detach() for k, v in latent.items()}
        policy_output = {"action": action, "logprob": logprob}
        return policy_output, (latent, action)

    def _train(self, data):
        metrics = {}

        with amp.autocast():
            post, context, mets = self._wm._train(data)
            metrics.update(mets)
            start = post
            reward = lambda f, s, a: self._wm.heads["reward"](
                self._wm.dynamics.get_feat(s)
            ).mode()
            metrics.update(self._task_behavior._train(start, reward)[-1])

            if self._config.expl_behavior != "greedy":
                mets = self._expl_behavior.train(start, context, data)[-1]
                metrics.update({"expl_" + key: value for key, value in mets.items()})

        for name, value in metrics.items():
            self._metrics.setdefault(name, []).append(value)