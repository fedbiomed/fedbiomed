from flask import request
import jsonschema


class Validator:
    def __init__(self, request):

        """
            Validator for reqeusted data. it is currently work
            for application/json type request. in feature it can be 
            extenden to other types of request such as form,url-encoded etc.
        """
        self._request   = request
        self._schema    = getattr(self, 'schema')
        self._type      = getattr(self, 'type') 

    def validate(self):
        
        """ Validation function for for provided schema currenly 
            it is only allowed to json schemas
        """
        if self._type == 'json':
            self._schema(request.json)
        else:
            raise Exception('Unspoorted validator type')



class JsonSchema(object):

    def __init__(self, schema, message: str = None):

        """ Schema class 

        Args: 
            schema (dict): A dictiory represent the valid schema 
                            for JSON object
            message (str): Message that will be return if the validation
                            is not successfull. If it is `None` it will return 
                            default `jsonschema` validation error message
        """

        self._schema = schema
        self._message = message

    def __call__(self, data):
        try:
            jsonschema.validate(data, self._schema)
        except jsonschema.ValidationError as e:
            if self._message:
                raise jsonschema.ValidationError(self._message)

            raise jsonschema.ValidationError(e.message)



class AddTabularData(Validator):
    
    """ Schema for reqeustion json data for adding new csv datasets """
    type   = 'json' 
    schema = JsonSchema({
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'path': {'type': 'string'},
                'tags': {'type': 'array' },
            },
            'required': ['name', 'path', 'tags']
        }, message = None)



