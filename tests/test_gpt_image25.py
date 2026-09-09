"""Offline regression tests for the GPT Image 2.5 update.

Run: python -m unittest discover -s tests -v
Requires the dependencies already used by the node: Pydantic 2, torch, numpy,
Pillow and requests. ComfyUI types and the operation transport are stubbed;
no real ComfyUI frontend, HTTP client, OpenAI account or image API is tested.
"""
import asyncio
import base64
import importlib
import io
import sys
import threading
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urljoin

import torch
from PIL import Image
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = '_gpt_image25_tests'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
comfy = types.ModuleType('comfy')
comfy.__path__ = []
typing = types.ModuleType('comfy.comfy_types.node_typing')
typing.IO = types.SimpleNamespace(**{k: k for k in ('STRING', 'COMBO', 'INT', 'IMAGE', 'MASK')})
typing.ComfyNodeABC = object
typing.InputTypeDict = dict
comfy_types = types.ModuleType('comfy.comfy_types')
comfy_types.__path__ = []
utils = types.ModuleType('comfy.utils')
def no_upscale(*args, **kwargs):
    raise AssertionError('Small test fixtures must not be resized')
utils.common_upscale = no_upscale

class HttpMethod(str, Enum):
    POST = 'POST'

class ApiEndpoint:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class SynchronousOperation:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def execute(self):
        raise AssertionError('Operation execution must be mocked in offline tests')

client = types.ModuleType(PACKAGE + '.apis.client')
client.ApiEndpoint = ApiEndpoint
client.HttpMethod = HttpMethod
client.SynchronousOperation = SynchronousOperation
with patch.dict(sys.modules, {
    'comfy': comfy, 'comfy.comfy_types': comfy_types,
    'comfy.comfy_types.node_typing': typing, 'comfy.utils': utils,
    PACKAGE + '.apis.client': client,
}):
    nodes = importlib.import_module(PACKAGE + '.nodes_api')
    schemas = importlib.import_module(PACKAGE + '.apis')

FLARE = 'gpt-image-2.5-flare'
SUNBURST = 'gpt-image-2.5-sunburst'
NEW = [FLARE, SUNBURST, FLARE + '-2026-09-08', SUNBURST + '-2026-09-08']
OLD = ['gpt-image-2', 'gpt-image-1.5', 'gpt-image-1', 'gpt-image-1-mini']
BASE = 'https://api.openai.com/v1'

def response():
    buf = io.BytesIO()
    Image.new('RGBA', (4, 4), (10, 20, 30, 0)).save(buf, format='PNG')
    return schemas.OpenAIImageGenerationResponse.model_validate({
        'data': [{'b64_json': base64.b64encode(buf.getvalue()).decode()}]
    })

def operation(model=FLARE, quality='high', background='opaque', images=None, mask=None, base=BASE):
    return nodes._build_operation('test', images or [], mask, quality, background,
        1, '1024x1024', 'auto', base, 'fake-test-key', model)

class Image25Tests(unittest.TestCase):
    def setUp(self):
        p = patch.object(nodes, 'read_user_config', return_value={})
        p.start()
        self.addCleanup(p.stop)

    def test_model_dropdowns(self):
        single = nodes.GPTImage1Generate.INPUT_TYPES()['optional']
        parallel = nodes.GPTImage2GenerateParallelX2.INPUT_TYPES()['optional']
        for widget in (single['model'], parallel['model_a'], parallel['model_b']):
            self.assertEqual(widget[1]['options'], NEW + OLD)
            self.assertEqual(widget[1]['default'], FLARE)

    def test_quality_dropdowns(self):
        single = nodes.GPTImage1Generate.INPUT_TYPES()['optional']
        parallel = nodes.GPTImage2GenerateParallelX2.INPUT_TYPES()['optional']
        for widget in (single['quality'], parallel['quality_a'], parallel['quality_b']):
            self.assertEqual(widget[1]['options'], ['auto','low','medium','high','xhigh','max'])
            self.assertEqual(widget[1]['default'], 'auto')

    def test_workflow_ids_outputs_and_input_order(self):
        self.assertEqual(set(nodes.NODE_CLASS_MAPPINGS), {'GPTImage2GenerateParallel', 'GPTImage2GenerateParallelX2'})
        self.assertEqual(nodes.GPTImage1Generate.RETURN_TYPES, ('IMAGE',))
        self.assertEqual(nodes.GPTImage2GenerateParallelX2.RETURN_NAMES, ('image_a','image_b'))
        self.assertEqual(list(nodes.GPTImage1Generate.INPUT_TYPES()['optional']), [
            'api_base','auth_token','model','seed','quality','background','size','n','image','mask','moderation'])
        expected = ['api_base','auth_token']
        for b in ('a','b'):
            expected += [f'image_{b}{i}' for i in range(1,6)]
            expected += [f'{n}_{b}' for n in ('mask','model','size','size_custom','quality','background','n','moderation')]
        self.assertEqual(list(nodes.GPTImage2GenerateParallelX2.INPUT_TYPES()['optional']), expected)

    def test_default_and_legacy_models(self):
        for model in (None, '', '  '):
            self.assertEqual(operation(model=model).request.model, FLARE)
        for model in OLD:
            self.assertEqual(operation(model=model).request.model, model)

    def test_all_new_qualities_serialize(self):
        for model in NEW:
            for quality in nodes.GPT_IMAGE_QUALITIES:
                for cls in (schemas.OpenAIImageEditRequest, schemas.OpenAIImageGenerationRequest):
                    with self.subTest(model=model, quality=quality, schema=cls.__name__):
                        data=cls(model=model,prompt='test',quality=quality).model_dump(mode='json',exclude_none=True)
                        self.assertEqual(data['quality'], quality)
                        self.assertEqual(data['model'], model)

    def test_old_qualities_still_serialize(self):
        for model in OLD:
            for quality in ('auto','low','medium','high'):
                self.assertEqual(operation(model=model,quality=quality).request.quality.value,quality)

    def test_incompatible_new_qualities_fail(self):
        for model in OLD:
            for quality in ('xhigh','max'):
                with self.assertRaisesRegex(ValueError,'requires GPT Image 2.5'):
                    operation(model=model,quality=quality)

    def test_invalid_generation_quality_fails(self):
        with self.assertRaises(ValidationError):
            schemas.OpenAIImageGenerationRequest(model=FLARE,prompt='test',quality='typo')

    def test_transparency_and_png(self):
        for model in NEW:
            data=operation(model=model,background='transparent').request.model_dump(mode='json')
            self.assertEqual(data['output_format'],'png')
            self.assertEqual(data['background'],'transparent')
        with self.assertRaisesRegex(ValueError,'does not support transparent'):
            operation(model='gpt-image-2',background='transparent')

    def test_api_path_join(self):
        for base in (BASE, BASE+'/', ' '+BASE+' ', 'https://example.test/proxy/v1'):
            op=operation(base=base)
            self.assertEqual(urljoin(op.api_base,op.endpoint.path),base.strip().rstrip('/')+'/images/generations')
        with self.assertRaisesRegex(ValueError,'Set api_base'):
            operation(base='')

    def test_generate_operation(self):
        op=operation(model=SUNBURST,quality='max')
        self.assertEqual(op.endpoint.path,'images/generations')
        self.assertIsNone(op.files)
        data=op.request.model_dump(mode='json',exclude_none=True)
        self.assertEqual(data['model'],SUNBURST)
        self.assertEqual(data['quality'],'max')
        self.assertNotIn('seed',data)

    def test_edit_multipart_and_mask(self):
        op=operation(model=SUNBURST,quality='xhigh',images=[torch.zeros(1,4,4,4)],mask=torch.ones(1,4,4))
        self.assertEqual(op.endpoint.path,'images/edits')
        self.assertEqual([name for name,_ in op.files],['image','mask'])
        for name,buf in op.files:
            buf.seek(0)
            self.assertEqual(Image.open(buf).mode,'RGBA')
        self.assertEqual(op.request.quality,'xhigh')

    def test_edit_multiple_references(self):
        op=operation(images=[torch.zeros(2,4,4,3), torch.zeros(1,4,4,3)])
        self.assertEqual([key for key,_ in op.files],['image[]']*3)

    def test_output_alpha(self):
        out=nodes.validate_and_cast_response(response())
        self.assertEqual(tuple(out.shape),(1,4,4,4))
        self.assertTrue(torch.equal(out[...,3],torch.zeros(1,4,4)))

    def test_single_node(self):
        seen=[]
        def execute(op):
            seen.append(op.request.model_dump(mode='json',exclude_none=True))
            return response()
        with patch.object(nodes.SynchronousOperation,'execute',new=execute):
            for model in (None, SUNBURST, 'gpt-image-2'):
                asyncio.run(nodes.GPTImage1Generate().api_call(prompt='test',model=model,seed=42,api_base=BASE,auth_token='test'))
        self.assertEqual([d['model'] for d in seen],[FLARE,SUNBURST,'gpt-image-2'])
        self.assertTrue(all('seed' not in d for d in seen))

    def test_parallel_overlap_and_independent_models(self):
        barrier=threading.Barrier(2,timeout=5)
        seen={}
        def execute(op):
            seen[op.request.prompt]=op.request.model
            barrier.wait()
            return response()
        with patch.object(nodes.SynchronousOperation,'execute',new=execute):
            result=asyncio.run(nodes.GPTImage2GenerateParallelX2().api_call('A','B',model_a=FLARE,model_b=SUNBURST,
                quality_a='max',quality_b='xhigh',api_base=BASE,auth_token='test'))
        self.assertEqual(seen,{'A':FLARE,'B':SUNBURST})
        self.assertEqual(len(result),2)

    def test_invalid_branch_no_execution(self):
        with patch.object(nodes.SynchronousOperation,'execute') as execute:
            with self.assertRaisesRegex(ValueError,'requires GPT Image 2.5'):
                asyncio.run(nodes.GPTImage2GenerateParallelX2().api_call('A','B',model_b='gpt-image-2',quality_b='max',api_base=BASE))
            execute.assert_not_called()

    def test_no_silent_model_fallback(self):
        seen=[]
        def fail(op):
            seen.append(op.request.model)
            raise RuntimeError('model_not_found')
        with patch.object(nodes.SynchronousOperation,'execute',new=fail):
            with self.assertRaisesRegex(RuntimeError,'model_not_found'):
                asyncio.run(nodes.GPTImage1Generate().api_call(prompt='test',model=SUNBURST,api_base=BASE))
        self.assertEqual(seen,[SUNBURST])

    def test_mask_without_image_clear_error(self):
        with self.assertRaisesRegex(Exception,'without an input image'):
            asyncio.run(nodes.GPTImage1Generate().api_call(prompt='test',mask=torch.zeros(1,4,4),api_base=BASE))

    def test_config_connection_and_custom_size(self):
        seen=[]
        def execute(op):
            seen.append((op.api_base,op.auth_token,op.request.size))
            return response()
        with patch.object(nodes,'read_user_config',return_value={'api_base':BASE,'auth_token':'test-key'}):
            with patch.object(nodes.SynchronousOperation,'execute',new=execute):
                asyncio.run(nodes.GPTImage2GenerateParallelX2().api_call('A','B',size_a='custom',size_custom_a='1792x1024'))
        self.assertEqual({row[2] for row in seen},{'1792x1024','auto'})
        self.assertTrue(all(row[:2]==(BASE+'/','test-key') for row in seen))

if __name__=='__main__':
    unittest.main()
