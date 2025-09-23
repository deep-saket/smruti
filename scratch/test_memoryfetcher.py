from modules.processing.MemoryFetcher import MemoryFetcher

class DummyMem:
    def __init__(self):
        self.data = {
            'semantic': {'Paris': 'Capital of France'},
            'procedural': {},
            'short_term': ['hi','hello','test'],
            'episodic': ['event1','event2'],
            'person': {'name':'Saket'}
        }
    def fetch(self, t, **kwargs):
        if t=='semantic':
            return self.data.get('semantic')
        if t=='procedural':
            return {}
        if t=='short_term':
            k=kwargs.get('k',3)
            return self.data.get('short_term')[:k]
        if t=='episodic':
            return self.data.get('episodic')
        if t=='person':
            return self.data.get('person')
        return None

print('Testing MemoryFetcher')
# Case 1: ner is None
mf = MemoryFetcher(DummyMem(), None)
res1 = mf(['Hello Paris','Who is John?'], user_id='u1')
print('Case1:', res1)

# Case 2: callable ner returning dict
def ner_callable(text, keys=None):
    return {'Location':['Paris']}
mf2 = MemoryFetcher(DummyMem(), ner_callable)
res2 = mf2(['I went to Paris yesterday'], user_id='u1')
print('Case2:', res2)

# Case 3: object with infer returning list of dicts
class GLStub:
    def infer(self, text, labels):
        return [{'type':'Person','entity':'John Doe'},{'type':'Location','entity':'Paris'}]
mf3 = MemoryFetcher(DummyMem(), GLStub())
res3 = mf3(['Talk about John in Paris'], user_id='u1')
print('Case3:', res3)

