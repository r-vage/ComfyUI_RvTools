# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from ..core import CATEGORY

class RvText_ReplaceStringV2:
    CATEGORY = CATEGORY.MAIN.value + CATEGORY.TEXT.value
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": "", "forceInput": False,"tooltip": "Input string to process."}),
                "regex": ("STRING", {"default": "", "tooltip": "Regular expression pattern to match."}),
                "replace_with": ("STRING", {"default": "", "tooltip": "Replacement string for matches."}),
                "remove_instructions": ("BOOLEAN", {"default": False, "forceInput": False, "tooltip": "When enabled, extract content from quotes at the start of the string, or if no quotes, remove everything before the first colon (:) including the colon itself."}),                                
                "list_select_first": ("BOOLEAN", {"default": False, "forceInput": False, "tooltip": "If enabled, extract the first numbered quoted choice (1.) from LLM output and use it as the result."}),
                "list_to_string": ("BOOLEAN", {"default": False, "forceInput": False, "tooltip": "If enabled, convert a numbered tips list into a single-line prompt and remove short labels (e.g., 'Lighting:')."}),
                "remove_background": ("BOOLEAN", {"default": False, "forceInput": False, "tooltip": "Whether to remove background description matches."}),
                "remove_subject": ("BOOLEAN", {"default": False, "forceInput": False, "tooltip": "Whether to remove subject description matches."}),
                "remove_subject_aggressive": ("BOOLEAN", {"default": False, "forceInput": False, "tooltip": "When enabled, remove pronoun-led subject clauses and possessive subject phrases (aggressive)."}),
                "remove_mood": ("BOOLEAN", {"default": False, "forceInput": False, "tooltip": "Whether to remove mood description matches."}),
                "remove_image": ("BOOLEAN", {"default": False, "forceInput": False, "tooltip": "Whether to remove image description matches."}),
                "cleanup": ("BOOLEAN", {"default": False, "forceInput": False, "tooltip": "When enabled, trim whitespace and remove surrounding quotes from the final output."}),
                
            }
        }

    def execute(
        self,
        string: str,
        regex: str,
        replace_with: str,
        remove_background: bool = False,
        remove_subject: bool = False,
        remove_subject_aggressive: bool = False,
        remove_mood: bool = False,
        remove_image: bool = False,
        cleanup: bool = False,
        list_select_first: bool = False,
        list_to_string: bool = False,
        remove_instructions: bool = False,
    ) -> tuple[str]:

        # Process string with regex replacement and optional description removals
        s = string or ""

        # Return unchanged if no operations requested
        try:
            no_toggles = not any([
                list_select_first,
                list_to_string,
                remove_background,
                remove_subject,
                remove_subject_aggressive,
                remove_mood,
                remove_image,
                remove_instructions,
            ])
        except Exception:
            no_toggles = True

        if no_toggles and not (regex and str(regex).strip()):
            return (s,)

        # Preprocessing steps
        try:
            if remove_instructions and s.strip():
                quote_match = re.match(r'^\s*["\']([^"\']*)["\']', s.strip())
                if quote_match:
                    s = quote_match.group(1)
                else:
                    colon_index = s.find(':')
                    if colon_index != -1:
                        s = s[colon_index + 1:].strip()

            if list_select_first and s.strip():
                m = re.search(r'(?s)^\s*1\.\s*(?:["\'])(.*?)(?:["\'])', s, flags=re.M)
                if m:
                    s = m.group(1)

            if list_to_string and s.strip():
                s = re.sub(r'(?s)^.*?(?=\d+\.)', '', s)  # remove header up to first numbered item
                s = re.sub(r'\*\*(.*?)\*\*', r'\1', s)  # remove bold markup
                s = re.sub(r'(?m)^\s*\d+\.\s*', '||', s)  # mark numbered items with delimiter
                s = re.sub(r'(?i)\b(?:lighting|composition|details|background|pose|makeup|props|editing|focus|storytelling)\s*:\s*', '', s)  # remove short label tokens
                s = re.sub(r'[\r\n\t]+', ' ', s)  # collapse newlines/tabs
                s = s.replace('||', ', ')  # replace delimiters with comma
                s = re.sub(r'^,\s+', '', s)  # clean leading comma
                s = re.sub(r'[ ]{2,}', ' ', s).strip()  # collapse extra spaces
        except Exception:
            pass

        if s.strip():
            try:
                # Regex patterns for description removal
                # Background: removes background/environment descriptions, stops before ", the" or ". the" to preserve "the overall" phrases
                background_pat = r"(?i)(?:(?:(?:the\s+)?backgrounds?|environment|setting|scene|surroundings|in the backgrounds?|in the environment|in the setting|in the scene|in the surroundings)\s*[:\-–]?\s*.*?(?=,\s+the|\.\s+the)|(?:[\.\?!,]\s*(?:The\s+)?(?:backgrounds?|environment|setting|scene|surroundings|in the background)\s+.*?(?=,\s+the|\.\s+the)))"
                # Subject: removes subject/person labels and descriptions
                subject_pat = r"(?i)(?:subject|person|people|man|woman|girl|boy|character)\s*[:\-–]?\s*[^\n\.;]+[\n\.;]?"
                # Mood: removes mood/atmosphere/vibe descriptions, including "overall" phrases, stops before ", the" or ". the" to preserve other "the overall" descriptions
                mood_pat = r"(?is)(?:\b(?:mood|moods|feeling|feelings|atmosphere|vibe|vibes|overall)\b\s*[:\-–]?\s*.*?(?=,\s+the|\.\s+the|[\n\.;]*$)|(?:^|[\.\?!,]\s*)(?:The\s+)?overall\s+.*?(?=,\s+the|\.\s+the|[\n\.;]*$)|(?:^|[\.\?!,]\s*)(?:The\s+)?(?:mood|moods|feeling|feelings|atmosphere|vibe|vibes)(?:\s+of(?:\s+the)?\s+(?:image|photograph|photo|scene|shot))?(?:\s+is|\s+are)?\s+.*?(?=,\s+the|\.\s+the|[\n\.;]*$))"
                # Image: removes image/photo labels and descriptions, avoids subject words like portrait/woman/man
                image_pat = r"(?i)(?:(?:\b(?:image|photo|photograph|picture|shot|render|illustration)\b)\s*(?:[:\-–]\s*|(?:is|was)\s+)(?![^\n\.;]{0,120}\b(?:portrait|woman|man|girl|boy|person|people|subject)\b)[^\n\.;]{1,200}[\n\.;]?)"

                def _preserve_lead(match):
                    lead = re.match(r'^\s*([\.\?!,])\s*', match.group(0))
                    if lead:
                        return lead.group(1) + ' '
                    return ''

                if remove_background:
                    s = re.sub(background_pat, _preserve_lead, s, flags=re.S)  # remove background descriptions
                if remove_subject:
                    subj_inner = r"(?:(?:subject|person|people|man|woman|girl|boy|character)|(?:he|she|him|her|they|them)|(?:young|old|elderly|teenage|middle-?aged|child|baby|adult))\b\s*[^\n\.;]*[\n\.;]?"
                    s = re.sub(r'(^|[\.\?!]\s)(?:The\s+)?' + subj_inner, r"\1", s, flags=re.S|re.I)  # remove subject sentences
                    s = re.sub(subject_pat, "", s, flags=re.S)  # remove subject fragments
                if remove_mood:
                    s = re.sub(mood_pat, _preserve_lead, s, flags=re.S)  # remove mood/atmosphere descriptions
                if remove_image:
                    s = re.sub(r'(?i)^[\s]*the\s+image\s+is\s+', '', s)  # remove "the image is" prefix
                    s = re.sub(r'(?i)^(?:.*?\b)?(?:close[- ]?up\s+portrait\s+of\s+|portrait\s+of\s+|headshot\s+of\s+)', '', s)  # remove portrait prefixes
                    s = re.sub(r'(?i)^[\s]*(?:a|an)\s+(?:[\w\-]+\s+)*(?:illustration|painting|drawing|sketch|photograph|photo)\s+(?:of\s+|featuring\s+)', '', s)  # remove "a [adjectives] illustration of/featuring"
                    s = re.sub(r'(?i)^[\s]*(?:a|an)\s+(?:[\w\-]+\s+)*(?:illustration|painting|drawing|sketch|photograph|photo)\s+in\s+(?:an?\s+)?[\w\s]+(?:style|art)\s*,?\s*featuring\s+', '', s)  # remove "a [adjectives] illustration in style, featuring"
                    image_inner = (
                        r"(?:\b(?:image|photo|photograph|picture|shot|render|illustration)\b)"
                        r"\s*(?:[:\-–]\s*|(?:is|was)\s+)"
                        r"(?![^\n\.;]{0,120}\b(?:portrait|woman|man|girl|boy|person|people|subject)\b)"
                        r"[^\n\.;]{1,200}[\n\.;]?"
                    )
                    s = re.sub(r'(^|[\.\?!]\s)'+image_inner, r'\1', s, flags=re.S|re.I)  # remove inline image descriptions
                if remove_subject_aggressive:
                    try:
                        pronoun_copula = re.compile(r'(?i)(^|[\.\?!]\s+)(?:The\s+)?\b(?:she|he|they|her|him|them|his|our|my)\b\s+(?:is|are|was|were|seems|appear(?:s)?|looks?)\s+', flags=re.S)
                        def _strip_pronoun_copula(m):
                            return m.group(1) or ''
                        s = pronoun_copula.sub(_strip_pronoun_copula, s)  # strip pronoun + copula
                    except Exception:
                        pass

                    pronoun_sentence_anchor = r'(^|[\.\?!]\s+)(?:The\s+)?(?:she|he|they|her|his|them|him)\b[^\n\.;]{0,200}[\n\.;]?'
                    s = re.sub(pronoun_sentence_anchor, _preserve_lead, s, flags=re.I|re.S)  # remove pronoun sentences
                    possessive_phrases = r"\b(?:her|his|their|my|our)\s+(?:face|eyes|hands|hair|skin|expression|eyebrows|mouth|nose|chin|cheeks|lips|teeth)\b[\w\s,\-]{0,80}"
                    s = re.sub(possessive_phrases, '', s, flags=re.I)  # remove possessive phrases
                    pronoun_sentence_any = r'(?<!\w)(?:she|he|they|her|him|them|his)\b[^\n\.;]{0,200}[\n\.;]?'
                    s = re.sub(pronoun_sentence_any, '', s, flags=re.I|re.S)  # remove pronoun fragments
            except Exception:
                pass

        # Apply user regex
        try:
            if regex and str(regex).strip():
                replaced = re.sub(regex, replace_with, s)  # apply custom regex replacement
            else:
                replaced = s
        except Exception:
            replaced = s

        # Optional cleanup
        if cleanup:
            replaced = re.sub(r"[\r\n]+", " ", replaced)  # normalize whitespace
            replaced = re.sub(r"[ ]{2,}", " ", replaced)  # collapse multiple spaces
            try:
                replaced = re.sub(r'\s*\.\s+(?=[a-z])', ' ', replaced)  # fix dangling periods
            except Exception:
                pass
            replaced = replaced.strip()  # remove leading/trailing whitespace
            replaced = replaced.replace('"', '')  # remove double quotes
            replaced = re.sub(r'\. ,\s*', '. ', replaced)  # fix ". ,"
            # Iteratively clean multiple consecutive punctuation marks throughout the string
            while re.search(r'[,.]\s*[,.]', replaced):
                replaced = re.sub(r'[,.]\s*[,.]', ',', replaced)  # collapse multiple punctuation to single comma
                replaced = re.sub(r',\s*,', ',', replaced)  # collapse comma-space-comma to comma
            # Clean trailing punctuation until only one dot remains
            while re.search(r'[,.]\s*$', replaced) or re.search(r'\.\s*\.', replaced):
                replaced = re.sub(r'[,.]\s*$', '', replaced).strip()  # remove trailing commas/periods
                replaced = re.sub(r'\.\s*\.', '.', replaced)  # collapse multiple periods
            # Remove all ending punctuation
            replaced = re.sub(r'[.,;:!?]+$', '', replaced).strip()
        return (replaced,)

NODE_NAME = 'Replace String v2 [Eclipse]'
NODE_DESC = 'Replace String v2'

NODE_CLASS_MAPPINGS = {
    NODE_NAME: RvText_ReplaceStringV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: NODE_DESC
}