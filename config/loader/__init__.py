import os
import dotenv
from config.loader.ConfigLoader import ConfigLoader

dotenv.load_dotenv()


project_root = os.environ.get("PROJECT_ROOT")
settings = ConfigLoader(os.path.join(project_root, "config/files/settings.yml")).get_config()
agent = ConfigLoader(os.path.join(project_root, "config/files/agent.yml")).get_config()