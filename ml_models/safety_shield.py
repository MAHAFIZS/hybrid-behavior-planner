# Action IDs
STOP, GO_FWD, TL, TR, SLOW = 0, 1, 2, 3, 4

class SafetyShield:
    def __init__(self, hard_stop=0.9, slow_zone=1.8):
        self.hard_stop = float(hard_stop)
        self.slow_zone = float(slow_zone)

    def override(self, proposed: int, obs) -> tuple[int, bool]:
        # obs = [v, front, left, right, goal_dist, goal_sin, goal_cos]
        front = float(obs[1])
        left  = float(obs[2])
        right = float(obs[3])

        # imminent collision -> do not go forward
        if front < self.hard_stop:
            if proposed in (TL, TR):
                return proposed, False
            # choose a turn direction that seems more open
            return (TR if left < right else TL), True

        # close obstacle -> reduce speed
        if front < self.slow_zone:
            if proposed == GO_FWD:
                return SLOW, True

        return proposed, False
