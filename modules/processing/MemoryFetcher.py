from common import CallableComponent

class MemoryFetcher(CallableComponent):
    def __init__(self, memory_manager, ner_processor, n_context=5):
        self.mem = memory_manager
        self.ner = ner_processor
        self.n_context = n_context

    def __call__(self, conversation: list[str], user_id: str) -> dict:
        """
        Given the full conversation history (list of utterances, oldest→newest),
        and the current user_id, returns a dict of:
          - points_of_interest: List[str]
          - fetched_memory: Dict[memory_type, data]
        """
        # 1) Extract points of interest from last n turns
        recent = conversation[-self.n_context:]
        pois = set()
        for turn in recent:
            ents = self.ner(turn, keys=["Person","Location","Organization","Position"])
            for lst in ents.values():
                pois.update(lst)
        # also you could embed & cluster, or extract keywords…

        # 2) Query each memory type for those POIs
        fetched = {}
        for poi in pois:
            # semantic lookup
            fact = self.mem.fetch("semantic", key=poi)
            if fact:
                fetched.setdefault("semantic", {})[poi] = fact

            # procedural lookup
            steps = self.mem.fetch("procedural", task_name=poi)
            if steps:
                fetched.setdefault("procedural", {})[poi] = steps

        # 3) Always grab the last 3 short-term exchanges
        fetched["short_term"] = self.mem.fetch("short_term", k=3)

        # 4) Grab last event if any
        episodes = self.mem.fetch("episodic", user_id=user_id)
        if episodes:
            fetched["episodic"] = episodes[-1]

        # 5) Grab user profile
        profile = self.mem.fetch("person", user_id=user_id)
        if profile:
            fetched["person"] = profile

        return {
            "points_of_interest": list(pois),
            "fetched_memory": fetched
        }