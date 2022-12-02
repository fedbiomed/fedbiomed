import jsonschema
from flask import request
from jsonschema import Draft7Validator, validators

# Constant validators settings
datasetName = {'type': 'string', "minLength": 4, "maxLength": 128,
               'errorMessages': {
                   'minLength': 'Dataset name must have at least 4 character',
                   'maxLength': 'Dataset name must be max 128 character'
               }}

datasetTags = {'type': 'array', "minItems": 1, "maxItems": 4,
               'errorMessages': {
                   'minItems': 'At least 1 tag should be provided',
                   'maxItems': 'Tags can be max. 4',
                   'type': 'Tags are in wrong format'
               }}

datasetDesc = {'type': 'string', "minLength": 4, "maxLength": 256,
               'errorMessages': {
                   'minLength': 'Description must have at least 4 character',
                   'maxLength': 'Description must be max 256 character'
               }}


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
                if "requiredMessages" in self._schema:
                    index = [i for i, val in enumerate(e.validator_value) if val in e.message][0]
                    if e.validator_value[index] in self._schema["requiredMessages"]:
                        message = self._schema["requiredMessages"][e.validator_value[index]]
                    else:
                        message = 'Please make sure all required fields have been filled'
                else:
                    message = 'Please make sure all required fields have been filled'

            elif e.relative_schema_path[0] == 'properties':
                field = e.relative_schema_path[1]
                reason = e.relative_schema_path[2]

                if field in self._schema['properties'] and \
                        'errorMessages' in self._schema['properties'][field] and \
                        reason in self._schema['properties'][field]['errorMessages']:
                    message = str(self._schema['properties'][field]['errorMessages'][reason]).format(e.instance)

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


class ListTrainingPlanRequest(Validator):
    type = 'json'
    schema = JsonSchema(
        {'type': "object",
         "properties": {
             "sort_by": {'type': ['string', 'null']},
             "select_status": {'type': 'string'},
             "search": {'type': ['object', 'null'],
                        'properties': {
                            "by": {"type": "string"},
                            "text": {"type": "string"}
                        },
                        "required": ["by", "text"]}

         },
         "required": []
         },
    )


class ApproveRejectTrainingPlanRequest(Validator):
    type = 'json'
    schema = JsonSchema(
        {'type': "object",
         "properties": {
             "training_plan_id": {"type": "string",
                                  "minLength": 1,
                                  'errorMessages': {"minLength": "model_id must have at least one character"},
                                  },
             "notes": {"type": ["string", "null"], "default": "No notes available"}
         },
         "required": ["training_plan_id"]
         }
    )


class DeleteTrainingPlanRequest(Validator):
    type = 'json'
    schema = JsonSchema(
        {'type': "object",
         "properties": {
             "training_plan_id": {"type": "string",
                                  "minLength": 1,
                                  'errorMessages': {"minLength": "model_id must have at least one character"},
                                  }
         },
         "required": ["training_plan_id"]
         }
    )


class TrainingPlanPreviewRequest(Validator):
    type = 'json'
    schema = JsonSchema(
        {'type': "object",
         "properties": {
             "training_plan_id": {"type": "string",
                                  "minLength": 1,
                                  'errorMessages': {"minLength": "model_path must have at least one character"},
                                  },
             "training_plan_path": {"type": "string"},
         },
         "required": ["training_plan_id"]
         }
    )


class AddDataSetRequest(Validator):
    """ Json Schema for reqeust of adding new datasets """

    type = 'json'
    schema = JsonSchema({
        'type': 'object',
        'properties': {
            'name': datasetName,
            'path': {'type': 'array'},
            'tags': datasetTags,
            'type': {'type': 'string',
                     'oneOf': [{"enum": ['csv', 'images']}],
                     'errorMessages': {
                         'oneOf': ' "%s" dataset type is not supported'
                     }},
            'desc': datasetDesc
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
        "properties": {"name": datasetName,
                       'path': {'type': 'array'},
                       'tags': datasetTags,
                       'type': {'type': 'string', 'default': "default",
                                'oneOf': [{"enum": ['default']}]},
                       'desc': datasetDesc

                       },
        "required": []
    })


class GetCsvData(Validator):
    type = 'json'
    schema = JsonSchema({
        "type": "object",
        "properties": {"path": {"type": "array"}},
        "required": ["path"]
    })


class ReadDataLoadingPlan(Validator):
    type = 'json'
    schema = JsonSchema({
        "type": "object",
        "properties": {"dlp_id": {"type": "string"}},
        "required": ["dlp_id"]
    })


class ValidateMedicalFolderReferenceCSV(Validator):
    type = 'json'
    schema = JsonSchema({
        "type": "object",
        "properties": {
            "reference_csv_path": {
                "type": "array",
                "errorMessages": {
                    "type": "CSV path should be given as an array"
                },
            },
            "medical_folder_root": {
                "type": "array",
                "errorMessages": {
                    "type": "ROOT path for MedicalFolder dataset should be given as an array."
                },
            },
            "index_col": {
                "type": "integer",
                "errorMessage": {
                    "type": "Index column should be an integer"}
            }
        },
        "required": ["reference_csv_path", "medical_folder_root", "index_col"]
    })


class ValidateMedicalFolderRoot(Validator):
    type = 'json'
    schema = JsonSchema({
        "type": "object",
        "properties": {
            "medical_folder_root": {
                "type": "array",
                "errorMessages": {
                    "type": "ROOT path should be given as an array"
                },
            }
        },
        "required": ["medical_folder_root"]
    })


class ValidateSubjectsHasAllModalities(Validator):
    type = 'json'
    schema = JsonSchema({
        "type": "object",
        "properties": {
            "medical_folder_root": {
                "type": "array",
                "errorMessages": {
                    "type": "ROOT path should be given as an array"
                },
            },
            "modalities": {
                "type": "array",
                "errorMessages": {
                    "type": "Modalities should be given as an array"
                },
            },
            "reference_csv_path": {
                "type": ["array", "null"],
                "errorMessages": {
                    "type": "CSV path should be given as an array"
                },
            },
            "index_col": {
                "type": ["integer", "null"],
                "errorMessage": {
                    "type": "Index column should be an integer"
                },
            },
            "dlp_id": {
                "type": ["string", "null"],
                "errorMessage": {
                    "type": "Data loading plan id should be a string"
                },
            },
        },
        "required": ["medical_folder_root", "modalities", "reference_csv_path", "index_col", "dlp_id"]
    })


class ValidateDataLoadingPlanDeleteRequest(Validator):
    type = 'json'
    schema = JsonSchema({
        "type": "object",
        "properties": {
            "dlp_id": {
                "type": "string",
                "errorMessages": {
                    "type": "Data loading plan ID should be given as a string"
                },
            },
        },
        "required": ["dlp_id"]
    })


class ValidateDataLoadingPlanAddRequest(Validator):
    type = 'json'
    schema = JsonSchema({
        "type": "object",
        "properties": {
            "modalities_mapping": {
                "type": "object",
                "errorMessages": {
                    "type": "DLP modalities mapping should be given as an object"
                },
            },
            "name": {
                "type": "string",
                "errorMessages": {
                    "type": "DLP name should be given as an array"
                },
            },
        },
        "required": ["modalities_mapping", "name"]
    })


class ValidateMedicalFolderAddRequest(Validator):
    type = 'json'
    schema = JsonSchema({
        "type": "object",
        "properties": {
            "medical_folder_root": {
                "type": "array",
                "errorMessages": {
                    "type": "ROOT path should be given as an array"
                },
            },
            "reference_csv_path": {
                "type": ["array", "null"],
                "default": None,
                "errorMessages": {
                    "type": "Reference CSV path should be given as an array"
                },
            },
            "index_col": {
                "type": ["integer", "null"],
                "default": None,
                "errorMessage": {
                    "type": "Index column should be declared as an integer"}
            },
            "dlp_id": {
                "type": ["string", "null"],
                "errorMessages": {
                    "type": "Data loading plan ID should be given as a string"
                },
            },
            'name': datasetName,
            'tags': datasetTags,
            'desc': datasetDesc
        },
        "required": ["medical_folder_root", "name", "tags", "desc", "dlp_id"]
    })


class ValidateLoginRequest(Validator):
    type = "json"
    schema = JsonSchema({
        'type': 'object',
        'properties': {
            'email': {
                'type': 'string',
                'minLength': 1,
                'errorMessages': {
                    'minLength': 'Email is missing'
                }
            },
            'password': {'type': 'string',
                         'minLength': 1,
                         'errorMessages': {
                             'minLength': 'Password is missing'
                         }},
        },
        'required': ['email', 'password'],
        'requiredMessages': {
            'email': 'E-mail is required!',
            'password': 'Password is required!'

        }
    })


class ValidateUserFormRequest(Validator):
    """ Json Schema for user registration and login requests"""
    type = 'json'
    schema = JsonSchema({
        'type': 'object',
        'properties': {
            'email': {
                'type': 'string',
                'minLength': 1,
                'errorMessages': {
                    'minLength': 'Email is missing'
                }
            },
            'password': {'type': 'string',
                         'minLength': 1,
                         'errorMessages': {
                             'minLength': 'Password is missing',
                             'type': 'Please make sure password respects required format'
                         }},
            'confirm': {'type': 'string',
                        'minLength': 1,
                        'errorMessages': {
                            'minLength': 'Password is missing',
                            'type': 'Please make sure password confirmation corresponds required format'
                        }},
            'old_password': {'type': 'string',
                             'minLength': 1,
                             'errorMessages': {
                                 'minLength': 'Old Password is missing',
                                 'type': 'Please make sure your provided old password'
                             }},
            'name': {'type': 'string',
                     'minLength': 1,
                     'errorMessages': {
                         'minLength': 'Name is missing'
                     }

                     },
            'surname': {'type': 'string',
                        'minLength': 1,
                        'errorMessages': {
                            'minLength': 'Surname is missing and it is required'
                        }

                        }
        },
        'required': ['email', 'password'],
        'requiredMessages': {
            'email': 'E-mail is missing!',
            'password': 'Password is missing!',
            'name': 'Password is missing!',
            'surname': 'Password is missing!',
        }
    })


class ValidateUserRemoveRequest(Validator):
    """ Json Schema for user registration and login requests"""
    type = 'json'
    schema = JsonSchema({
        'type': 'object',
        'properties': {
            'user_id': {
                'type': 'string',
                'minLength': 1,
                'errorMessages': {
                    'minLength': 'Missing ID for for to delete'
                }
            },
        },
        'required': ['user_id'],
        'requiredMessages': {
            'user_id': 'ID is required to remove user',
        }
    })


class ValidateUserChangeRoleRequest(Validator):
    """ Json Schema for changing user role"""
    type = 'json'
    schema = JsonSchema({
        'type': 'object',
        'properties': {
            'user_id': {
                'type': 'string',
                'minLength': 1,
                'errorMessages': {
                    'minLength': 'Missing ID for user'
                }
            },
            'role': {
                'type': 'integer',
                'enum': [1, 2],
                'errorMessages': {
                    'enum': 'Invalid role'
                }
            },
        },
        'required': ['user_id', 'role'],
        'requiredMessages': {
            'user_id': 'ID is required to change role',
            'role': 'User role is required to change user role',
        }
    })


class ListUserRegistrationRequest(Validator):
    type = 'json'
    schema = JsonSchema({
        'type': 'object',
        "properties": {
            "search": {'type': 'string'}
        },
        "required": []
    })


class ValidateAdminRequestAction(Validator):
    """ Json Schema for administrator actions to handle registration requests"""
    type = 'json'
    schema = JsonSchema({
        'type': 'object',
        'properties': {
            'request_id': {
                'type': 'string',
                'description': 'id of the user request to validate',
            }
        },
        'required': ['request_id'],
    }, message=None)
