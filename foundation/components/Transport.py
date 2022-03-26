

from foundation.base.base_component import (
    BaseComponent,
    component_registry,
)

#Build an industry
@component_registry.add
class Transport(BaseComponent):

    name = "Transport"
    component_type = "Transport"
    required_entities = []
    agent_subclasses = ["BasicMobileAgent"]


    def __init__(self, *base_component_args, payment=10, **base_component_kwargs):
        super().__init__(*base_component_args, **base_component_kwargs)
        self.payment = payment



    def component_step(self):
      return []
    def generate_masks(self):
      return []
    def generate_observations(self):
      return []
    def get_additional_state_fields(self, agent_cls_name):
      return {}
    def get_n_actions(self, agent_cls_name):
      return []

