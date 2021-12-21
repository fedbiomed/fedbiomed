from flask import request
import jsonschema
from jsonschema import Draft7Validator, validators


def extend_validator(validator_class):
    """ Extending json validator to set default values
        if it is specified in the schema

        @source: https://readthedocs.org/projects/python-jsonschema/downloads/pdf/latest/
    """

    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):

        """ Setting default values for requested dict object to validate """

        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(validator, properties,
                                         instance, schema):
            yield error

    return validators.extend(
        validator_class, {"properties": set_defaults},
    )


# Extend Draft7Validator
JsonBaseValidator = extend_validator(Draft7Validator)


class Validator:

    def __init__(self, request):
        """
            Validator for requested data. it is currently work
            for application/json type request. in the future, it can be
            extended to other types of request such as form,url-encoded etc.

            Args:
                request (flask.request): Request data
        """

        self._request = request
        self._schema = getattr(self, 'schema')
        self._type = getattr(self, 'type')

    def validate(self):
        """ Validation function for provided schema. Currently,
            it is only allowed to control json schemas
        """

        if self._type == 'json':
            self._schema(request.json)
        else:
            raise Exception('Unsupported schema validator type')


class JsonSchema(object):

    def __init__(self, schema, message: str = None):
        """Schema class 

        Args:
            schema (dict): A dictionary represent the valid schema
                            for JSON object
            message (str): Message that will be return if the validation
                            is not successful. If it is `None` it will return
                            default `jsonschema` validation error message
        """

        self._schema = schema
        self._message = message

    def __call__(self, data):
        """
            Validating schema while calling the schema object
            with given data.

            Args:
                data (dict): Request.json, which comes as dict object.
                            This data will be validated
        """

        try:
            JsonBaseValidator(self._schema).validate(data)
        except jsonschema.ValidationError as e:
            if self._message:
                raise jsonschema.ValidationError(self._message)
            raise jsonschema.ValidationError(e.message)


class AddDataSetRequest(Validator):
    """ Json Schema for reqeust of adding new datasets """

    type = 'json'
    schema = JsonSchema({
        'type': 'object',
        'properties': {
            'name': {'type': 'string', "minLength": 4, "maxLength": 128},
            'path': {'type': 'array'},
            'tags': {'type': 'array',  "minItems": 1, "maxItems": 4},
            'type': {'type': 'string',
                     'oneOf': [{"enum": ['csv', 'images']}]},
            'desc': {'type': 'string'}
        },
        'required': ['name', 'path', 'tags', 'desc', 'type']
    }, message=None)


class ListDataFolder(Validator):
    """  JSON schema for request of /api/repository/list """

    type = 'json'
    schema = JsonSchema({
        'type': "object",
        "properties": {
            "path": {'type': 'array', 'default': []}
        },
        "required": []
    })


class RemoveDatasetRequest(Validator):
    """  JSON schema for request of /api/datasets/remove """

    type = 'json'
    schema = JsonSchema({
        'type': "object",
        "properties": {
            "dataset_id": {'type': 'string'}
        },
        "required": ["dataset_id"]
    })


class PreviewDatasetRequest(Validator):
    """  JSON schema for request of /api/datasets/preview """

    type = 'json'
    schema = JsonSchema({
        'type': "object",
        "properties": {
            "dataset_id": {'type': 'string', "minLength": 4, "maxLength": 254}
        },
        "required": ["dataset_id"]
    })


class UpdateDatasetRequest(Validator):
    """  JSON schema for request of /api/datasets/update """

    type = 'json'
    schema = JsonSchema({
        'type': "object",
        "properties": {"name": {'type': 'string', "minLength": 4, "maxLength": 128},
                       "dataset_id": {'type': 'string'},
                       "path": {'type': 'array'},
                       "tags": {'type': 'array', "minItems": 1, "maxItems": 4},
                       "desc": {'type': 'string', "minLength": 4, "maxLength": 254}},
        "required": ["dataset_id", "tags", "desc"]
    })


class AddDefaultDatasetRequest(Validator):

    """ JSON schema for validating request for adding new
        default dataset
    """

    type = 'json'
    schema = JsonSchema({
        'type': "object",
        "properties": {"name": {'type': 'string',
                                "default": 'mnist', "minLength": 4, "maxLength": 128}},
        "required": []
    })
