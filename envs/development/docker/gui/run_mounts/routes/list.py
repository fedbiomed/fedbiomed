from . import api 


@api.route('/hello' , methods=['GET'])
def hello_api():

    return 'HELLO API'