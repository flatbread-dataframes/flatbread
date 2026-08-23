from flatbread.output.html.constants import FLATBREAD_TABLE_URL

def on_page_context(context, page, config, nav):
    context["flatbread_table_url"] = FLATBREAD_TABLE_URL
    return context
