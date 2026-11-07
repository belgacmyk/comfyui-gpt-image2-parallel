"""Real request schemas + real HTTP adapter; requests.request is mocked."""
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from test_gpt_image25 import schemas

path=Path(__file__).resolve().parents[1]/'apis/client.py'
spec=importlib.util.spec_from_file_location('_image26_real_client',path)
transport=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=transport
spec.loader.exec_module(transport)

class TransportTests(unittest.TestCase):
    def test_real_json_and_multipart_fields(self):
        for edited in (False,True):
            cls=schemas.OpenAIImageEditRequest if edited else schemas.OpenAIImageGenerationRequest
            request=cls(prompt='synthetic transport test',model='gpt-image-2.6-flare',
                        output_format='gif',complexity='medium',frame_count=7,frame_duration_ms=200)
            files=[('image',io.BytesIO(b'not uploaded: mocked transport'))] if edited else None
            operation=transport.SynchronousOperation(
                endpoint=transport.ApiEndpoint(path='images/edits' if edited else 'images/generations',
                    method=transport.HttpMethod.POST,request_model=cls,response_model=schemas.OpenAIImageGenerationResponse),
                request=request,files=files,api_base='https://api.openai.com/v1/',auth_token='fake-test-key')
            with patch.object(transport.requests,'request') as send:
                send.return_value.content=b'{}'
                send.return_value.json.return_value={'data':[{'b64_json':'ZmFrZQ=='}]}
                operation.execute()
            kwargs=send.call_args.kwargs
            payload=kwargs['data' if edited else 'json']
            self.assertEqual(payload['output_format'],'gif')
            self.assertEqual(payload['complexity'],'medium')
            self.assertEqual(payload['frame_count'],7)
            self.assertTrue(kwargs['verify'])
            self.assertEqual(kwargs['url'],'https://api.openai.com/v1/images/'+('edits' if edited else 'generations'))
            self.assertEqual('Content-Type' in kwargs['headers'],not edited)

if __name__=='__main__':unittest.main()
