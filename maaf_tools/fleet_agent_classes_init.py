
def create_fleet_agent_classes_dict(dir: str):
    import robot_language as rl
    import os

    dict_from_rl = {}
    rl.tool.set_verbosity(2)

    for rl_file in os.listdir(dir):
        rl_path = os.path.join(dir, rl_file)
        if os.path.isfile(rl_path):
            if os.path.splitext(rl_path)[1] == ".rl":
                rl.tool.start_phase("Parsing")
                _, model_instance = rl.parse_file(
                    rl_path)
                if rl.tool.is_ok:
                    rl.tool.start_phase("Resolve")
                    model_instance.resolve()
                dict_from_rl[model_instance.skillsets[0].name] = {
                    "specs": {},
                    "skillsets": [skill.name for skill in model_instance.skillsets[0].skills]
                }

    return dict_from_rl


if __name__ == "__main__":
    import sys
    import json
    target_path = sys.argv[1]
    with open("test.json", "w") as j_file:
        json.dump(create_fleet_agent_classes_dict(
            target_path), fp=j_file, indent=4)
