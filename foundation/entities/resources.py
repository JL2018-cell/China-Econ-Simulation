# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from foundation.base.registrar import Registry


class Resource:
    """Base class for Resource entity classes.

    Resource classes describe entities that can be a part of an agent's inventory.

    Resources can also be a part of the world as collectible entities: for each
    Resource class with Resource.collectible=True, a complementary
    ResourceSourceBlock Landmark class will be created in landmarks.py. For each
    collectible resource in the environment, the world map will include a resource
    source block channel (representing landmarks where collectible resources are
    generated) and a resource channel (representing locations where collectible
    resources have generated).
    """

    name = None
    color = None  # array of RGB values [0 - 1]
    collectible = None  # Is this something that exists in the world?
    # (versus something that can only be owned)

    def __init__(self):
        assert self.name is not None
        assert self.color is not None
        assert self.collectible is not None


resource_registry = Registry(Resource)


#@resource_registry.add
#class Wood(Resource):
#    """Wood resource. collectible."""
#
#    name = "Wood"
#    color = np.array([107, 143, 113]) / 255.0
#    collectible = True


@resource_registry.add
class Agriculture(Resource):
    """Agriculture industry"""

    name = "Agriculture"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True

@resource_registry.add
class Minerals(Resource):
    """Minerals industry"""

    name = "Minerals"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True

@resource_registry.add
class Energy(Resource):
    """Minerals industry"""

    name = "Energy"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True


@resource_registry.add
class Tourism(Resource):
    """Minerals industry"""

    name = "Tourism"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True


@resource_registry.add
class IT(Resource):
    """Minerals industry"""

    name = "IT"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True


@resource_registry.add
class Finance(Resource):
    """Minerals industry"""

    name = "Finance"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True


@resource_registry.add
class Manufacturing(Resource):
    """Minerals industry"""

    name = "Manufacturing"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True

@resource_registry.add
class Construction(Resource):
    """Minerals industry"""

    name = "Construction"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True

@resource_registry.add
class Transport(Resource):
    """Minerals industry"""

    name = "Transport"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True

@resource_registry.add
class Retail(Resource):
    """Minerals industry"""

    name = "Retail"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True


@resource_registry.add
class Education(Resource):
    """Minerals industry"""

    name = "Education"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True

