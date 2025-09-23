import importlib.util
from pathlib import Path

# caminho absoluto para o 01_json_to_df.py
module_path = Path(__file__).resolve().parent / "01_json_to_df.py"
spec = importlib.util.spec_from_file_location("json_to_df_module", module_path)
json_to_df_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(json_to_df_module)

# expõe as funções publicamente
load_dict_json_flat = json_to_df_module.load_dict_json_flat
load_prospects = json_to_df_module.load_prospects