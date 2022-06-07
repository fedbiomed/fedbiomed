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

            # Raise custom error messages
            message = None
            if e.relative_schema_path[0] == 'required':
                message = 'Please make sure all required fields has been filled'
            elif e.relative_schema_path[0] == 'properties':
                field = e.relative_schema_path[1]
                reason = e.relative_schema_path[2]

                if field in self._schema['properties'] and \
                        'errorMessages' in self._schema['properties'][field] and \
                        reason in self._schema['properties'][field]['errorMessages']:
                    message = self._schema['properties'][field]['errorMessages'][reason] % e.instance

            if message:
                raise jsonschema.ValidationError(message)
            else:
                raise jsonschema.ValidationError(e.message)


class ListDatasetRequest(Validator):
    type = 'json'
    schema = JsonSchema({
        'type': "object",
        "properties": {
            "search": {'type': 'string'}
        },
        "required": []
    })


class ListModelRequest(Validator):
    type = 'json'
    schema = JsonSchema(
        {'type': "object",
         "properties": {
             "sort_by": {'type': 'string'},
             "select_status": {'type': 'string'}
         },
         "required": []
         },
        )


class ApproveRejectModelRequest(Validator):
    type = 'json'
    schema = JsonSchema(
        {'type': "object",
         "properties": {
             "model_id": {"type": "string",
                          "minLength": 1,
                          'errorMessages': {"minLength": "model_id must have at least one character"},
                          },
             "notes": {"type": "string"}
         },
         "required": ["model_id"]
         }
    )


class DeleteModelRequest(Validator):
    type = 'json'
    schema = JsonSchema(
        {'type': "object",
         "properties": {
             "model_id": {"type": "string",
                          "minLength": 1,
                          'errorMessages': {"minLength": "model_id must have at least one character"},
                          }
         },
         "required": ["model_id"]
         }
    )


class ModelPreviewRequest(Validator):
    type = 'json'
    schema = JsonSchema(
        {'type': "object",
         "properties": {
             "model_path": {"type": "string",
                            "minLength": 1,
                            'errorMessages': {"minLength": "model_path must have at least one character"},
            
                            },
             "model_id": {"type": "string"},
         },
         "required": []  
         }
    )


class AddDataSetRequest(Validator):
    """ Json Schema for reqeust of adding new datasets """

    type = 'json'
    schema = JsonSchema({
        'type': 'object',
        'properties': {
            'name': {'type': 'string', "minLength": 4, "maxLength": 128,
                     'errorMessages': {
                         'minLength': 'Dataset name must have at least 4 character',
                         'maxLength': 'Dataset name must be max 128 character'
                     }
                     },
            'path': {'type': 'array'},
            'tags': {'type': 'array', "minItems": 1, "maxItems": 4,
                     'errorMessages': {
                         'minItems': 'At least 1 tag should be provided',
                         'maxItems': 'Tags can be max. 4',
                         'type': 'Tags are in wrong format'
                     }
                     },
            'type': {'type': 'string',
                     'oneOf': [{"enum": ['csv', 'images']}],
                     'errorMessages': {
                        'oneOf': ' "%s" dataset type is not supported'
                     }},
            'desc': {'type': 'string', "minLength": 4, "maxLength": 256,
                     'errorMessages': {
                         'minLength': 'Description must have at least 4 character',
                         'maxLength': 'Description must be max 256 character'
                     }}
        },
        'required': ['name', 'path', 'tags', 'desc', 'type'],
    }, message=None)


class ListDataFolder(Validator):
    """  JSON schema for request of /api/repository/list """

    type = 'json'
    schema = JsonSchema({
        'type': "object",
        "properties": {
            "path": {'type': 'array', 'default': []},
            "refresh": {'type': 'boolean', 'default': False}
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
            "dataset_id": {'type': 'string', "minLength": 4, "maxLength": 256}
        },
        "required": ["dataset_id"]
    })


class UpdateDatasetRequest(Validator):
    """  JSON schema for request of /api/datasets/update """

    type = 'json'
    schema = JsonSchema({
        'type': "object",
        "properties": {"name": {'type': 'string', "minLength": 4, "maxLength": 128,
                                'errorMessages': {
                                    'minLength': 'Dataset name must have at least 4 character',
                                    'maxLength': 'Dataset name must be max 128 character'
                                }},
                       "dataset_id": {'type': 'string'},
                       "path": {'type': 'array'},
                       "tags": {'type': 'array', "minItems": 1, "maxItems": 4},
                       "desc": {'type': 'string', "minLength": 4, "maxLength": 256,
                                "errorMessages": {
                                    "minLength": 'Description must have at least 4 character',
                                    "maxLength": 'Description must be max 256 character'
                                }},
                       },
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
                                "default": 'mnist', "minLength": 4, "maxLength": 128,
                                'errorMessages': {
                                    'minLength': 'Dataset name must have at least 4 character',
                                    'maxLength': 'Dataset name must be max 128 character'
                                    }
                                },
                       'path': {'type': 'array'},
                       'tags': {'type': 'array', "minItems": 1, "maxItems": 4, 'default': ["#MNIST", "#dataset"],
                                'errorMessages': {
                                    'minItems': 'At least 1 tag should be provided',
                                    'maxItems': 'Tags can be max. 4',
                                    'type': 'Tags is in wrong format'
                                }},
                       'type': {'type': 'string', 'default': "default",
                                'oneOf': [{"enum": ['default']}]},
                       'desc': {'type': 'string', "minLength": 4, "maxLength": 256, 'default' : "Default MNIST dataset",
                                'errorMessages': {
                                    'minLength': 'Description must have at least 4 character',
                                    'maxLength': 'Description must be max 256 character'
                                }}

                       },
        "required": []
    })
