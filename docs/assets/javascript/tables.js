app.document$.subscribe(function() {
  var dfTables = document.querySelectorAll("table.dataframe");
  dfTables.forEach(function(table) {
    table.removeAttribute("class");;
    table.removeAttribute("border");;
  })

  // Apply style to markdown tables
  var tables = document.querySelectorAll("article table")
  tables.forEach(function(table) {
    new Tablesort(table)
  })
})