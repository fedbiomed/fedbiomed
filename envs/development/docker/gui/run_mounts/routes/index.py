from app import app


@app.route('/' , methods=['GET'])
def index():
    
    """ The index route. This route should render the 
        react build files. Which is located athe fron-end folder.
    """
    return 'HELLO FEDBIOMED'

@app.route('/static' , methods=['GET'])
def static_react():
    
    """ Static route is the route for static files of the
        react front-end applicaltion
    """

    return 'HELLO STATIC FILES'



