import json
import unittest
import warnings

try:
    from maaf_tools.datastructures.organisation.MOISEPlus.MoiseModel import MoiseModel
except ImportError:
    from maaf_tools.maaf_tools.datastructures.organisation.MOISEPlus.MoiseModel import MoiseModel

with open("icare_alloc_config/icare_alloc_config/__cache/__CoHoMa_organisation_model_v1_moise+_model.json",
          "r") as file:
    TEST_MODEL_DICT = json.load(file)

class TestMoiseModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Suppress warnings during tests (if desired)
        warnings.simplefilter("ignore", category=UserWarning)

    def setUp(self):
        with open("icare_alloc_config/icare_alloc_config/__cache/__CoHoMa_organisation_model_v1_moise+_model.json",
                  "r") as file:
            model = json.load(file)

        # Create a MoiseModel instance from the test JSON dictionary before each test
        self.model = MoiseModel.from_dict(item_dict=TEST_MODEL_DICT)

        print(self.model)

    def test_model_initialization(self):
        # Check that the model has been instantiated properly.
        self.assertIsInstance(self.model, MoiseModel)
        # Assert number of roles, role_relations, and groups
        self.assertEqual(len(self.model.structural_specification["roles"]), 21)
        self.assertEqual(len(self.model.structural_specification["role_relations"]), 14)
        self.assertEqual(len(self.model.structural_specification["groups"]), 5)

    def test_check_model_definition(self):
        # The provided model should be valid.
        is_valid = self.model.check_model_definition(stop_at_first_error=True, verbose=0)
        self.assertTrue(is_valid, "The model definition should pass validation.")

    def test_goal_skill_requirements(self):
        # For goal "g_point_obs", the required skill is "goto"
        skills = self.model.get_goal_skill_requirements("g_point_obs", verbose=0)
        self.assertEqual(set(skills), {"goto"})

    def test_agent_goal_compatibility(self):
        # An agent with the "goto" skill should be compatible with goal "g_point_obs"
        self.assertTrue(self.model.check_agent_goal_compatibility(["goto"], "g_point_obs"))
        # An agent without the required skill should not be compatible.
        self.assertFalse(self.model.check_agent_goal_compatibility([], "g_point_obs"))

    def test_agent_mission_compatibility(self):
        # Mission m_scouting has goals "g_point_obs", "g_axis_obs", "g_zone_obs", all requiring "goto"
        self.assertTrue(self.model.check_agent_mission_compatibility(["goto"], "m_scouting"))
        self.assertFalse(self.model.check_agent_mission_compatibility([], "m_scouting"))

    def test_agent_role_compatibility(self):
        # Role "Scout" is associated with mission "m_scouting" (via obligations)
        # and thus requires the "goto" skill.
        self.assertTrue(self.model.check_agent_role_compatibility(["goto"], "Scout"))
        self.assertFalse(self.model.check_agent_role_compatibility([], "Scout"))

    def test_get_goals_associated_with_role(self):
        # For role "Scout", we expect the goals of the m_scouting mission to appear.
        goals = self.model.get_goals_associated_with_role("Scout", names_only=True)
        # The m_scouting mission defines: ["g_point_obs", "g_axis_obs", "g_zone_obs"]
        self.assertIn("g_point_obs", goals)
        self.assertIn("g_axis_obs", goals)
        self.assertIn("g_zone_obs", goals)

    def test_get_roles_compatible_with_skillset(self):
        # Define an agent skillset containing all skills needed to execute any mission.
        skillset = ["goto", "trap", "track", "neutralise"]
        roles = self.model.get_roles_compatible_with_skillset(skillset)
        # At minimum, roles tied to missions requiring these skills (from obligations) should be included.
        for role in ["Scout", "Monitor", "Patroller", "Trapper", "Tracker", "Neutraliser"]:
            self.assertIn(role, roles)

    def test_merge_models(self):
        # Create a duplicate model and merge it with the existing one.
        model_copy = MoiseModel.from_dict(item_dict=TEST_MODEL_DICT)
        result = self.model.merge(model_copy, prioritise_local=True)
        self.assertTrue(result, "Merging with a duplicate model should succeed.")
        # After merging, the model should still pass validation.
        self.assertTrue(self.model.check_model_definition(stop_at_first_error=True, verbose=0))

    def test_serialization_roundtrip(self):
        # Test the conversion to a dictionary and back to a model.
        model_dict = self.model.asdict()
        model_from_dict = MoiseModel.from_dict(model_dict)
        # Compare their string representations as a proxy for equality.
        self.assertEqual(str(self.model), str(model_from_dict),
                         "Model serialization followed by deserialization should return the same model representation.")


if __name__ == "__main__":
    unittest.main()
