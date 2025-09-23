from common import CallableComponent

class MemoryFetcher(CallableComponent):
    def __init__(self, memory_manager, ner_processor, n_context=5):
        super().__init__()
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

        if self.ner is None:
            # use logger from parent CallableComponent
            self.logger.debug("No NER processor available; skipping entity extraction.")
        else:
            # support both: callable processors (ner(text, keys=...)) and objects with `infer(text, labels)`
            labels = ["Person", "Location", "Organization", "Position"]
            for turn in recent:
                try:
                    if callable(self.ner):
                        # try calling with the expected kw arg name used by NERProcessor
                        try:
                            ents = self.ner(turn, keys=labels)
                        except TypeError:
                            # fallback to positional args or different signature
                            try:
                                ents = self.ner(turn, labels)
                            except Exception:
                                # as a last resort try `infer`
                                ents = getattr(self.ner, "infer")(turn, labels)
                    else:
                        # object with an infer method (e.g. GLiNERInfer)
                        infer = getattr(self.ner, "infer", None)
                        if infer is None:
                            self.logger.warning("NER processor provided but has no callable interface: %s", type(self.ner))
                            continue
                        ents = infer(turn, labels)

                    # ents is expected to be a dict (NERProcessor) or list (GLiNERInfer)
                    if isinstance(ents, dict):
                        for lst in ents.values():
                            pois.update(lst)
                    elif isinstance(ents, list):
                        # GLiNERInfer returns list[dict] with keys 'type' and 'entity'
                        for ent in ents:
                            etype = ent.get("type") or ent.get("entity_type") or ent.get("label")
                            val = ent.get("entity") or ent.get("text") or ent.get("value")
                            if etype in labels and val:
                                pois.add(val)
                    else:
                        self.logger.debug("NER returned unsupported type %s; skipping", type(ents))
                except Exception as e:
                    self.logger.exception("NER extraction failed for turn: %s", e)
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