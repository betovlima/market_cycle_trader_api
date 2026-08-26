class RocDecisionPolicyError(RuntimeError):
    pass


class RocDecisionPolicyNotFound(RocDecisionPolicyError):
    pass


class RocDecisionPolicyConflict(RocDecisionPolicyError):
    pass
