let pathname = window.location.pathname;
let url      = window.location.href;
let origin   = window.location.origin;

let doc_paths = ['/getting-started', '/tutorials', '/user-guide', '/developer']
let is_version_available
let deprecated_versions = ["v4.3", "v4.2", "v4.1", "v4.", "v3.5", "v3.4", "v3.3", "v3.2", "v3.2"]
/**
 * Check current page is documentation
 * @returns {boolean}
 */
const check_is_docs = () => {
    if( pathname.startsWith('/getting-started') || pathname.startsWith('/tutorials') ||
        pathname.startsWith('/user-guide') || pathname.startsWith('/developer/') ) {
        return true
    }
    return false
}

let is_docs = check_is_docs()

/**
 * Joins base and given relative path
 * @param {string} base 
 * @param {string} path 
 * @returns 
 */
function joinUrl (base, path) {
    if (path.substring(0, 1) === "/") {
      // path starts with `/`. Thus it is absolute.
      return path;
    }
    if (base.substring(base.length-1) === "/") {
      // base ends with `/`
      return base + path;
    }
    return base + "/" + path;
  }


  var getAbsoluteUrl = (function() {
    var a;
    return function(url) {
        if(!a) a = document.createElement('a');
        a.href = url;
        return a.pathname;
    }
    })();


/**
 * Do redirection if doc url does not start with version number
 */
if(is_docs) {
    // This is necessary to get version JSON. Otherwise, it means that doc is served under development
    $.getJSON('/versions.json', function(data) {
        if(data){
            window.stop()
            let v = data.versions
            let url = '/latest' + pathname
            url = url.replace('//' , '/')
            window.location.replace(url);
        }
    })
}

/**
 * When document is ready
 */
$(document).ready(async function(){

    let versions;
    let docurl;


    /**
     * Gets versions JSON object and run version related functions
     */
    try {
        console.log(versions_json)
        if(!versions_json){
            versions_json="/versions.json"
        }

        let data = await $.getJSON(versions_json)
        versions = data.versions
        docurl = data.docurl
        const selectBox = versionSelectBox(versions);
        $('.sidebar-inner').prepend(
            '<div class="version-box"> Version: '+selectBox+'</div></hr>'
        )
        updateVersionBox()
        displayVersionWarning()
        updateHomeDocsURLS()

        is_version_available = true
    }catch {
        is_version_available = false
        console.warn('Versioning wont be available')
    }


    /**
     * This function create popup if user is not displaying
     * the latest version of fedbiomed.
     */
    function displayVersionWarning(){
        let ver_url = checkPathHasVersion(pathname)
        if(ver_url != false && ver_url != 'latest'){
            let body = $('body')
            let navDesk = $('.container-fluid.top-bar')
            let navMob = $('.container-fluid.top-bar-mobile')
            let main = $('.main')
            let url = pathname.replace(ver_url, 'latest').replace('//','/')
            if(body.attr('id') === 'not-found'){
                body.prepend(
                    '<div class="version-warning">' +
                    '<a href="'+url+'">'+
                    'This documentation is not available for version "' + ver_url + '" click here to display it on latest version' +
                    '</a>' +
                    '</div>'
                )
            }else{
                body.prepend(
                    '<div class="version-warning">' +
                    '<a href="'+url+'">'+
                    'You are not displaying the latest version of Fed-BioMed. Please click here to display the latest version of this documentation' +
                    '</a>' +
                    '</div>'
                )
            }

            navDesk.addClass('warning')
            navMob.addClass('warning')
            if( $('.top-bar-mobile').css('display') === 'block'){
                main.addClass('warning')
            }else{
                main.addClass('warning-desktop')
            }
        }
    }

    /**
     * This function creates version select box
     * @param versions {object}
     * @returns {string}
     */
    function versionSelectBox(versions){
        let select = '<select name="version" id="version">'
        Object.keys(versions).forEach( (item) =>{
            let l = item === 'latest' ? 'Latest ' : ''
            let urlValue = item === 'latest' ? 'latest' : versions[item]
            select += '<option value="'+ urlValue + '">' + l + versions[item] +'</option>'
        });
        select += '</select>';
        return select;
    }

    /**
     * This function check whether URL has version tag if it has, it returns the version tag.
     * Otherwise return false. That mean latest version of doc is displayed.
     * @param path
     * @returns {string|boolean}
     */
    function checkPathHasVersion(path) {

        let  ver = Object.assign({}, versions);
        for(let i=0; i<Object.keys(ver).length; i++){
            let key = Object.keys(ver)[i]
            let vers = ver[key]
            if(path.startsWith('/latest') || path.includes('/latest/') ){
                return 'latest'
            }else if( path.startsWith('/' + vers.toString()) || path.includes('/' +  vers.toString() + '/') ){
                return vers.toString()
            }
        }

        return false

    }


    /**
     * This function gets displayed doc version and updates version select box
     */
    function updateVersionBox(){
        // Update selected box selected option
        let ver_url = checkPathHasVersion(pathname)
        if(ver_url !== false){
            $('#version').val(ver_url)
        }
    }

    /**
     * Event handler for version select box action. Redirect pages based on
     * selected version tag
     */
    $(document).on('change','#version', function() {

        let version = $(this).val();
        let v = checkPathHasVersion(pathname)
        let abs_url = getAbsoluteUrl(base_url)
        let version_url

        if( deprecated_versions.includes(version) ){
            version_url = abs_url.replace(v, version) 
            window.location.replace( version_url);
        }else{
            if (!v){
                alert("Can not display chosen version.")
            }else{
                let abs_url = getAbsoluteUrl(base_url)
                let location = pathname.replace(abs_url, '')
                let version_url = abs_url.replace(v, version) + location
                window.location.replace( version_url);
            }
        }

    })


    // -------------------------------------------------------------------------------------

    $(document).on('click' , '.has-sub' , function(e){
        let child = $(this).children("ul")
        if(child.hasClass('active')){
            child.removeClass('active');
        }else{
            child.addClass('active');
        }
    })

    // Mobile menu switcher ---------------------------------------------------------------------------------------------------

    $('.hum-menu').on('click', function(){
        let navMobile = $('nav.top-mobile')
        if(navMobile.hasClass('active')){
            $('.hum-menu .open').css({'display': 'block'})
            $('.hum-menu .close').css({'display': 'none'})
            navMobile.removeClass('active');
            navMobile.animate({'max-height' : 0}, 100)
        }else{
            navMobile.addClass('active');
            navMobile.animate({'max-height': '800px'}, 100)
            $('.hum-menu .open').css({'display': 'none'})
            $('.hum-menu .close').css({'display': 'block'})
        }
    });


    // Doc Sidebar open actions ------------------------------------------------------------------------------------------------------
    $(".sidebar-menu-left li.has-sub-side div").on('click', function(){
        let ul = $(this).siblings("ul")
        if(ul.hasClass('active')){
            ul.removeClass('active')
            ul.css({'display' : 'none'})
        }else{
            ul.addClass('active')
            ul.css({'display' : 'block'})
        }


    });

    var pathArray = window.location.pathname.split('/');
    let hash = pathArray[-1]

    let tocChange = function(){
        var scrollTop = $(document).scrollTop();
        var anchorsH2 = $('body').find('h2');
        var anchorsH3 = $('body').find('h3');
        var anchorsH4 = $('body').find('h4');
        var anchorsH5 = $('body').find('h5');

        let anchors = []
        for (var i = 0; i < anchorsH2.length; i++){
            anchors.push(anchorsH2[i])
        }

        for (var i = 0; i < anchorsH3.length; i++){
            anchors.push(anchorsH3[i])
        }

        for (var i = 0; i < anchorsH4.length; i++){
            anchors.push(anchorsH4[i])
        }

        for (var i = 0; i < anchorsH5.length; i++){
            anchors.push(anchorsH5[i])
        }

        for (var i = 0; i < anchors.length; i++){
            if (scrollTop > $(anchors[i]).offset().top - 200 && scrollTop < $(anchors[i]).offset().top + $(anchors[i]).height() + 200) {
                $('.sidebar-right nav ul li a').removeClass('active');
                $('.sidebar-right nav ul li a[href="#' + $(anchors[i]).attr('id') + '"]').addClass('active');
            }
        }
    }

    // When windows is loaded
    tocChange()

    // Onscroll
    $(window).scroll(function(){
        tocChange()
    });

    // On ToC item click
    $('.sidebar-right nav ul li a').on('click',function(){
        tocChange()
        let div = $(this).attr('href')
        div = document.getElementById(div.replace('#', ''))
        $('html, body').animate({
            scrollTop: $(div).offset().top - 100
        }, 500);

    })

    $('a.autorefs.autorefs-internal').on('click', function(){
        tocChange()
        let div = $(this).attr('href')
        div = document.getElementById(div.replace('#', ''))
        $('html, body').animate({
            scrollTop: $(div).offset().top - 100
        }, 500);
    })

    $('a.scroller-link').on('click',function(){
        tocChange()
        let div = $(this).attr('href')
        $('html, body').animate({
            scrollTop: $(div).offset().top - 150
        }, 500);
    })

    // Search Box -------------------------------------------------------------------------------
    let search_box = '<div class="search-box">\n' +
                    '    <div class="search-overlay"></div>\n' +
                    '    <form autocomplete="off" id="search-form" action="?q=:name">\n' +
                    '    <input autocomplete="false" name="hidden" type="text" style="display:none;">' +
                    '        <input id="search" name="search" type="text" placeholder="search"/>\n' +
                    '        <label>\n' +
                    '            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M9.5 3A6.5 6.5 0 0 ' +
                    '1 16 9.5c0 1.61-.59 3.09-1.56 4.23l.27.27h.79l5 5-1.5 1.5-5-5v-.79l-.27-.27A6.516 6.516 0 0 1 9.5 16 6.5 6.5 ' +
                    '0 0 1 3 9.5 6.5 6.5 0 0 1 9.5 3m0 2C7 5 5 7 5 9.5S7 14 9.5 14 14 12 14 9.5 12 5 9.5 5Z"></path></svg>\n' +
                    '        </label>\n' +
                    '    </form>\n' +
                    '    <div class="search-result-wrapper">\n' +
                    '        <div class="loader"><div class="lds-ring"><div></div><div></div><div></div><div></div></div></div>\n' +
                    '        <ul class="search-results" id="search-result-list"></ul>\n' +
                    '    </div>\n' +
                    '</div>'

    // Append search box.
    $('.top-bar').append(search_box)

    // Open wide search panel
    $("#search").focusin(function(e){
        $(".search-box").addClass('active')
    })

    // Close wide search panel
    $(".search-overlay").on('click', function(e){
        $(".search-box").removeClass('active')
    })

    // Search Worker Actions -----------------------------------------------------------------------------------------

    // Retrieve correct version index
    let version_status = checkPathHasVersion(pathname)
    let version = version_status !== false ? version_status : ''
    console.log('Current version: ', version )

    // Indexing
    const SearchWorker = new Worker(search_worker_js);
    SearchWorker.postMessage({type: 'INDEX', payload: {search_index_json: search_index_json}})

    // Search worker event listener
    SearchWorker.onmessage = (message) => {
        switch (message.data.type) {
            case "SEARCH_RESULT":
                render_search_result(message.data.payload)
                display_search_loader(false)
            default:
                break;
        }
    }

    //setup before functions
    let typingTimer;                //timer identifier
    let doneTypingInterval = 1000;  //time in ms, 5 seconds for example
    let input = $('#search');

    //on keyup, start the countdown
    input.on('keyup', function () {
      clearTimeout(typingTimer);
      typingTimer = setTimeout(searchRequest, doneTypingInterval);
    });

    //on keydown, clear the countdown
    input.on('keydown', function () {
      clearTimeout(typingTimer);
    });

    // On search form is submitted
    $("#search-form").on('submit', function(e){
        e.preventDefault()
        display_search_loader(true)
        searchRequest()
    })

    /**
     * Handles search request
     */
    function searchRequest () {
        let search_text = input.val()

        if(search_text){
             display_search_loader(true)
            SearchWorker.postMessage({type:'SEARCH', payload:search_text})
        }else{
            clear_search_result()
        }
    }

    /**
     * Displays loader while loading search results
     * @param state
     */
    const display_search_loader = (state) => {
        if(state){
            $(".search-box .loader").addClass('active')
        }else{
            $(".search-box .loader").removeClass('active')
        }
    }

    /**
     * Renders search results
     * @param data
     */
    const render_search_result = (data) => {
        let search_list = create_search_result_list(data)
        $("#search-result-list").empty().append(search_list)
    }

    /**
     * Clears search results
     * @param data
     */
    const clear_search_result = (data) => {
        $("#search-result-list").empty()
    }


    /**
     * Create HTML list element for search results
     * @param data
     * @returns {string}
     */
    const create_search_result_list = (data) => {
        let content = ""
        let version_suffix
        let is_doc_url

        let version_status = checkPathHasVersion(pathname)
        let version = version_status !== false ? version_status+'/' : is_version_available ? 'latest/' : ''

        if(data.length > 0){
            data.forEach(result => {
                let raw_location = "/"+ result.location
                is_doc_url = doc_paths.some((path) => { return raw_location.startsWith(path)})
                version_suffix = is_doc_url ? '/' + version : '/'
                content += '<li><a href="'+ joinUrl(base_url, result.location)+'"><h4>'+result.title+'</h4><p>'+
                    result.text.substring(0, 100) + "...</p></a></li>"
            })
        }else{
            content = '<p>No result found.</p>'
        }
        return content
    }
})


