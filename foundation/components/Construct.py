#Refer to build.py

from foundation.base.base_component import (
    BaseComponent,
    component_registry,
)

#Build an industry
@component_registry.add
class Construct(BaseComponent):

    name = "Construct"
    component_type = "Construct"
    required_entities = ["Agriculture", "Minerals"]
    agent_subclasses = ["BasicMobileAgent"]


    def __init__(self, *base_component_args, payment=10, **base_component_kwargs):
        super().__init__(*base_component_args, **base_component_kwargs)
        self.payment = payment
