"""Apply the requested clickable repository link formatting to the report."""
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

def format_repository_link(document):
    url = 'https://github.com/YuvrajC7/sms-spam-detection'
    for paragraph in document.paragraphs:
        if paragraph.text == 'Project repository\n' + url:
            paragraph.clear()
            paragraph.add_run('Project repository\n')
            hyperlink = OxmlElement('w:hyperlink')
            hyperlink.set(qn('r:id'), paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True))
            hyperlink.set(qn('w:history'), '1')
            run = OxmlElement('w:r')
            properties = OxmlElement('w:rPr')
            properties.append(OxmlElement('w:b'))
            colour = OxmlElement('w:color'); colour.set(qn('w:val'), '0000FF')
            properties.append(colour)
            underline = OxmlElement('w:u'); underline.set(qn('w:val'), 'single')
            properties.append(underline)
            run.append(properties)
            text = OxmlElement('w:t'); text.text = url
            run.append(text); hyperlink.append(run); paragraph._p.append(hyperlink)
            return
    raise ValueError('Project repository paragraph not found')
