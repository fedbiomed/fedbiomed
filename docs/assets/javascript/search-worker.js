importScripts('./lunr.js')

/**
 * Search index URL
 * @type {string}
 */

let SEARCH_INDEX = "waiting"
let INDEX_CONTENT = "waiting"
let stop = true
let maxResults = 50
let searchWaitTime = 1000
let searchTimer


/**
 * Event listener for the `postMessage` comes from employer of the worker
 */
self.addEventListener("message", async (e) => {
    switch (e.data.type) {
        case "SEARCH":
            stop = false
            search(e.data.payload)
            break;
        case "INDEX":
            try {
                INDEX_CONTENT = await fetch_search_index(e.data.payload.search_index_json)
                SEARCH_INDEX = await create_search_index(INDEX_CONTENT)
            } catch {
                INDEX_CONTENT = false
                console.error('Can not get search index')
            }
            break;
        case "STOP":
            stop = true
        default:
            console.warning('No action the take')
    }
})


/**
 * Gets search_index.json
 * @param version
 * @returns {Promise<unknown>}
 */
const fetch_search_index = (search_index_json) => {
    return new Promise( (resolve, reject) => {
        fetch(search_index_json)
            .then(response => {
                return(response.json())
            })
            .then(data => {
                resolve(data.docs)
            })
            .catch(error => {
                console.error("Unable to get search index json. Search functionality might not work", error)
                reject(false)
            })
    })
}

/**
 * Creates search index using lunr.js by parsing search_index.json
 * @param search_index
 * @returns {lunr}
 */
const create_search_index = (search_index) => {

    let search_idx = lunr(function () {
        this.ref('location')
        this.field('title')
        this.field('text')
        Object.keys(search_index).forEach(function (key) {
            this.add(search_index[key])
        }, this)
    });

    return search_idx
}

/**
 * Does searching
 * @param text
 */
const search = (text) => {
    if(SEARCH_INDEX){
        if(SEARCH_INDEX === "waiting"){
            this.postMessage({type:"LOADING" , payload: true})
            clearTimeout(searchTimer)
            console.log('Waiting to load search indexes')
            searchTimer = setTimeout( () => search(text), searchWaitTime);
        }else{
            clearTimeout(searchTimer)
            let result = SEARCH_INDEX.search(text)
            parse_content(result)
        }
    }else{
        console.error("Can not perform search something is wrong")
    }

}

/**
 * Parse content of search_index.json based on found search result
 * @param content
 */
const parse_content = (content) => {
    let results = []
    let max = content.length <= maxResults ? content.length : maxResults
    if(content){
        for(let i=0; i<max; i++) {
            let ref = INDEX_CONTENT.filter((entry) => entry.location === content[i].ref)
            results = results.concat(ref)
        }
    }else{
        console.log("Empty result")
    }
    this.postMessage({type: 'SEARCH_RESULT' , payload: results})
}

/**
 * Gets already parsed lunr index json if it exists
 * TODO: Did not implemented correctly. Build script should create lunr index json
 * @param version
 * @returns {Promise<unknown>}
 */
const fetch_lunr_search_index = (version) => {
    return new Promise((resolve, reject) => {
        let v = version !== '' ? '/' + version : ''
        fetch(v + '/search/search_lunr_index.json')
            .then(response => {
                 if (response.ok) {
                     resolve(response.json())
                 }else{
                     throw new Error('Can not get `search index for lunr');
                 }
            })
            .then(data => {
                SEARCH_INDEX = lunr.Index.load(data)
                resolve(SEARCH_INDEX)
            })
            .catch(error => {
                console.warn("No search index found. Search index will be created")
                reject(false)
            })
    })
}