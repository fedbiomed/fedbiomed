import React from 'react'
import styles from "./Inputs.module.css"

export const Label = (props) => {

    return (
        <label 
            className="input-label"
            htmlFor={props.for}
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

    const ref = React.useRef()
    return (
        <input 
            className="input"
            type={props.type}
            name={props.name}
            id={props.id}
            ref={ref}
            onChange={props.onChange}
            onKeyDown={props.onKeyDown}
            value={props.value}
            placeholder={props.placeholder}
            minLength={props.minlength}
        />
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
            ref  = {props.areaRef}
            onChange = {props.onChange}
            value = {props.value ? props.value : undefined}
            placeholder={props.placeholder}
            style={props.style}
        />
    )
}

/**
 * Tag Input Component
 * @param {*} props 
 * @returns 
 */
export const Tag = (props) => {

    const [currentTagText, setCurrentTagText] = React.useState("");
    const [tags, setTags] = React.useState(props.tags ? props.tags : []);
    const refInput = React.useRef()

    React.useEffect(() => {
        if(props.tags) {
            setTags(props.tags)
        }
    }, [props.tags])

    /**
     * On event click lıke space or enter get tag 
     * @param {HTMLInput} element 
     */
    const handleKeyDown = (element) => {
        
        if(tags.length <=3) {

            setCurrentTagText(element.target.value);
            if ( (element.keyCode === 13 && currentTagText && /\S/.test(currentTagText)) ||
                 (element.keyCode === 32 && currentTagText && /\S/.test(currentTagText)) ||
                 (element.keyCode === 9 && currentTagText && /\S/.test(currentTagText)) ) {
                let tags_update = [...tags, currentTagText.replace(/\s/g, '')]
                setTags((prevTags) => {return [...prevTags, currentTagText] });
                setCurrentTagText('');
                
                // Let parent componenet know that the 
                // tags has been chaged or new one added
                if(props.onTagsChange){
                    props.onTagsChange(props.name, tags_update)
                }
                if(props.onChange){
                    props.onChange({
                         target: {
                             name : props.name,
                             value : tags_update
                         }
                    })
                }
            }
        }
    };

    /**
     * When user click on input box this function
     * automatically focus to input field
     */
    const onInputClick = () => {
        refInput.current.focus()
    }

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
         if(props.onTagsChange){
            props.onTagsChange( props.name, newTagArray)
        }
         if(props.onChange){
             props.onChange({
                 target: {
                     name : props.name,
                     value : newTagArray
                 }
             })
         }
    };


    return (
        <div className="tags-input" onClick={onInputClick}>
          <div
            className="tags"
            style={{ display: tags.length > 0 ? "flex" : "none" }}
          >
            {tags.map((tag, index) => {
              return (
                      <div key={index} className="tag">
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
              ref  = {refInput}
              placeholder={props.placeholder}
            />
          </div>
        </div>
    )
}


/**
 * Select box component
 * @param {*} props 
 * @returns 
 */
export const Select = (props) => {

    return (
        <select 
            className={`select ${props.className}`}
            type = {props.type} 
            name = {props.name}
            id   = {props.id}
            ref = {props.selectRef}
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

export const CheckBox = (props) => {

    const [checked, setChecked] = React.useState(props.checked)

    const handleChange = () => {
        setChecked(!checked)
        props.onChange(!checked)
    }

    return(
        <label className={styles.checkboxLabel}>
            <input className={styles.checkboxInput} name={props.name} type={"checkbox"} checked={checked} onChange={handleChange}/>
                <span className={styles.checkboxText}>
                    {props.children}
                </span>
        </label>
    )
}