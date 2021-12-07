import React from 'react'


export const Label = (props) => {

    return (
        <label 
            className="input-label"
            for={props.for}  
        >
            {props.children}
        </label>
    )
}


/**
 * 
 * @param {*} props 
 * @returns 
 */
export const Text = (props) => {
    return (
        <input 
            className="input"
            type        = {props.type} 
            name        = {props.name}
            id          = {props.id}
            onChange    = {props.onChange}
            onKeyDown   = {props.onKeyDown}
            ></input>
    )
}

/**
 * 
 * @param {*} props 
 * @returns 
 */
export const TextArea = (props) => {
    return (
        <textarea
            className="input"
            type = {props.type} 
            name = {props.name}
            id   = {props.id}
            onChange = {props.onChange}
            value = {props.value ? props.value : null}
        ></textarea>
    )
}

/**
 * Tag Input Component
 * @param {*} props 
 * @returns 
 */
export const Tag = (props) => {

    const [currentTagText, setCurrentTagText] = React.useState("");
    const [tags, setTags] = React.useState([]);


    /**
     * On event click lıke space or enter get tag 
     * @param {HTMLInput} element 
     */
    const handleKeyDown = (element) => {
        
        if(tags.length <=3) {

            setCurrentTagText(element.target.value);
            if ( (element.keyCode === 13 && currentTagText) ||  (element.keyCode === 32 && currentTagText)  ) {
                let tags_update = [...tags, currentTagText]
                setTags((prevTags) => {return [...prevTags, currentTagText] });
                setCurrentTagText('');
                
                // Let parent componenet know that the 
                // tags has been chaged or new one added
                if(props.onTagsEntered){
                    props.onTagsEntered(tags_update)
                }
            }
        }



    };

    /**
     * When new tag text ıs entered
     * @param {HTMLInput} element 
     */
    const handleTagChange = (element) => {
        if(tags.length <=3) { 
            setCurrentTagText(element.target.value);
        }
    }

    /**
     * Remove tag by given index
     * @param {int} index 
     */
    const removeTag = (index) => {
        const newTagArray = tags;
        newTagArray.splice(index, 1);
        setTags([...newTagArray]);
    };


    return (
        <div className="tags-input">
          <div
            className="tags"
            style={{ display: tags.length > 0 ? "flex" : "none" }}
          >
            {tags.map((tag, index) => {
              return (
              <div className="tag">
                 <button
                    onClick={() => removeTag(index)}
                    className="close"
                  >X</button>
                  {tag}
              </div>
               )
            })}
          </div>
          <div className="input-field">
            <input
              type="text"
              onKeyDown={handleKeyDown}
              onChange={handleTagChange}
              value={currentTagText}
            />
          </div>
        </div>
    )
}


/**
 * 
 * @param {*} props 
 * @returns 
 */
export const Select = (props) => {
    
    return (
        <select 
            className="select"
            type = {props.type} 
            name = {props.name}
            id   = {props.id}
            onChange = {props.onChange}
        >

            { props.options ? (
                props.options.map( (item,key) => {
                    return(
                        <option key={key} value={item.value}>{item.name}</option>
                    )
                })
            ) : props.children ? (
                props.children
            ) : null}
        </select>
    )
}