from cocoa.core.scenario_db import Scenario as BaseScenario
from cocoa.core.schema import Attribute
from core.kb import KB

class Scenario(BaseScenario):
    def __init__(self, uuid, attributes, kbs):
        super(Scenario, self).__init__(uuid, attributes, kbs)

    @staticmethod
    def from_dict(schema, raw):
        if schema is not None:
            attributes = schema.attributes
        else:
            assert 'attributes' in raw
        if 'attributes' in raw:
            attributes = [Attribute.from_json(raw_attr) for raw_attr in raw['attributes']]
        return Scenario(raw['uuid'], attributes, [KB.from_dict(attributes, kb) for kb in raw['kbs']])

    def to_dict(self):
        d = super(Scenario, self).to_dict()
        return d