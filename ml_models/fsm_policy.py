# Action IDs must match:
STOP, GO_FWD, TL, TR, SLOW = 0, 1, 2, 3, 4

class SafetyLimits:
    def __init__(self, hard_stop=1.0, slow_zone=2.0):
        self.hard_stop = hard_stop
        self.slow_zone = slow_zone

def apply_safety(proposed: int, front: float, lim: SafetyLimits) -> int:
    if front < lim.hard_stop:
        if proposed in (TL, TR):
            return proposed
        return STOP
    if front < lim.slow_zone:
        if proposed == GO_FWD:
            return SLOW
        return proposed
    return proposed

class FsmConfig:
    def __init__(self, obstacle_trigger=2.5, obstacle_clear=3.0):
        self.obstacle_trigger = obstacle_trigger
        self.obstacle_clear = obstacle_clear

class FsmPolicy:
    FOLLOW, AVOID, STOPSTATE = 1, 2, 3

    def __init__(self):
        self.state = self.FOLLOW
        self.cfg = FsmConfig()
        self.safety = SafetyLimits()

    def act(self, obs):
        v, front, left, right, gd, goal_sin, goal_cos = obs

        # transitions
        if front < self.safety.hard_stop:
            self.state = self.STOPSTATE
        elif front < self.cfg.obstacle_trigger:
            self.state = self.AVOID
        elif self.state == self.AVOID and front > self.cfg.obstacle_clear:
            self.state = self.FOLLOW
        elif self.state == self.STOPSTATE and front > self.cfg.obstacle_clear:
            self.state = self.FOLLOW

        # propose
        if self.state == self.FOLLOW:
            if goal_sin > 0.15:
                proposed = TL
            elif goal_sin < -0.15:
                proposed = TR
            else:
                proposed = GO_FWD

        elif self.state == self.AVOID:
            proposed = TR if left < right else TL

        else:  # STOPSTATE escape turn
            proposed = TR if left < right else TL

        return apply_safety(proposed, float(front), self.safety)
