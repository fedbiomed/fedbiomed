


const eventBus = {
  
    on(event, callback) {
      // attachs an EventListener to the document object. The callback will be called when the event gets fired.
      document.addEventListener(event, (e) => callback(e.detail));
    },
    dispatch(event, data) {
      // fires an event using the CustomEvent API
      document.dispatchEvent(new CustomEvent(event, { detail: data }));
    },
    remove(event, callback) {
      // removes the attached event from the document object
      document.removeEventListener(event, callback);
    },
  };
  export default eventBus;