from urllib.parse import urlparse

from flatbread.output.html.constants import FLATBREAD_TABLE_URL

def on_page_context(context, page, config, nav):
    context["flatbread_table_url"] = FLATBREAD_TABLE_URL
    return context

def on_page_content(html, config, **kwargs):
    site_url = config.get("site_url", "/")
    base_path = urlparse(site_url).path
    return html.replace('src="/assets/', f'src="{base_path}assets/')
