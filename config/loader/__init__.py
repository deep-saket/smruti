import os
import dotenv
from config.loader.ConfigLoader import ConfigLoader

dotenv.load_dotenv()


project_root = os.environ.get("PROJECT_ROOT")
settings = ConfigLoader(os.path.join(project_root, "config/files/settings.yml")).get_config()
agent = ConfigLoader(os.path.join(project_root, "config/files/agent.yml")).get_config()

settings['db']['audio_recogniser'] = settings['db']['audio_recogniser'].format(project_root = project_root)
settings['memory']['dir'] = settings['memory']['dir'].format(project_root = project_root)
