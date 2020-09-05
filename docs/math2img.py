"""Auto-conversion of LaTeX math expression inside a markdown file

Idea is taken from vscode:extension/MeowTeam.vscode-math-to-image.
The difference is that math expressions must be inside markdown-link '[text](url)' or
markdown-image '![text](url)' tags.

When in markdown-link tag, the result will be seen while typing, with a LaTeX math viewer
extension, like vscode:extension/goessner.mdmath. This allows to see the result while typing,
by just removing the '!' sign. The result from math2img URL, will be seen after '!' is added.
"""
import sys
import glob
import re
import urllib.parse

GENERATE_URLS = True    # If False, all math URLs will be deleted

# Match strings in format ...[$$-latex-math-$$](-url-)...
# then replace -url- with an URL encoded -latex-math-
PATTERN = re.compile(r'\[\$\$(?P<math>[^\$]*)\$\$\]\((?P<url>[^\)]*)\)')
URL_PREFIX='https://render.githubusercontent.com/render/math?math='

total_matches = 0
total_changes = 0

def encode_math(match):
    global total_matches, total_changes
    total_matches += 1

    # Check for malformed URL: The brackets are currently not supported
    math = match.group('math')
    if '(' in match.group('url'):
        print('Warning: Existing URL for expression "%s" contain "(" symbol(s)'%math, file=sys.stderr)
        return match.group(0)

    # Build the LaTeX math encoded URL
    if GENERATE_URLS:
        url_result = URL_PREFIX + urllib.parse.quote(math)
    else:
        url_result = ''

    if match.group('url') == url_result:
        return match.group(0)

    # Locate existing URL
    span = match.span('url')
    span = slice(span[0] - match.start(0), span[1] - match.start(0))
    # Replace URL
    res = [*match.group(0)]
    res[span] = url_result

    total_changes += 1
    return ''.join(res)

def main(files):
    for fn in files:
        print('Processing', fn, '...')
        global total_matches, total_changes
        total_matches, total_changes = 0, 0

        # Read the process the file
        with open(fn) as fd:
            result = ''
            for ln in fd:
                result += PATTERN.sub(encode_math, ln)

        # Overwrite the file content
        if total_changes:
            with open(fn, 'w') as fd:
                fd.write(result)

        if total_matches:
            print('  Matches', total_matches, ', changes', total_changes)
        else:
            print('Warning: No LaTeX math expressions found', file=sys.stderr)

    return 0

if __name__ == '__main__':
    files = sys.argv[1:]
    # Check for delete URLs option
    if files[0] == '-d':
        GENERATE_URLS = False
        files = files[1:]

    if not files:
        files = glob.glob('*.md')

    ret = main(files)
    if ret:
        sys.exit(ret)
