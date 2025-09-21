import os, sys, importlib, traceback
print('PYTHON:', sys.executable)
print('PYTHON_VERSION:', sys.version.replace('\n',' '))
print('PROJECT_ROOT env:', os.environ.get('PROJECT_ROOT'))
print('PYTHONPATH env:', os.environ.get('PYTHONPATH'))
print('sys.path[:5]:', sys.path[:5])
print('cwd:', os.getcwd())
try:
    cl = importlib.import_module('config.loader')
    print('config.loader file:', getattr(cl, '__file__', None))
    print('settings keys:', sorted(list(getattr(cl, 'settings', {}).keys())))
    print('agent keys:', sorted(list(getattr(cl, 'agent', {}).keys())))
    mcp = importlib.import_module('src.MCPProcessor')
    print('MCPProcessor class:', mcp.MCPProcessor.__name__)
    print('IMPORT_OK')
except Exception:
    traceback.print_exc()
    print('IMPORT_FAIL')

