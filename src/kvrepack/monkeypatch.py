from importlib.metadata import version
import os
import transformers

def replace_qwen(method):
    if method == "asymkv_official":
        from kvrepack.asymkv_official.integration import install_qwen2_5_vl_asymkv_official

        install_qwen2_5_vl_asymkv_official()
        return

    if method == 'snapkv':
        print("Using SnapKV!")
        from .qwen_model import qwen_flash_attn2_forward_SnapKV
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_SnapKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        from .qwen_model import qwen_flash_attn2_forward_SparseMM, adakv_qwen_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = adakv_qwen_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_SparseMM

    elif method == "sparsemm_kvrepack":
        print("Using SparseMM-KVRepack!")
        from .qwen_model import adakv_qwen_forward
        from .qwen_model_sparsemm_kvrepack import qwen_flash_attn2_forward_SparseMMKVRepack
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = adakv_qwen_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_SparseMMKVRepack

    elif method == "lookm":
        print("Using LookM!")
        from .qwen_model import qwen_flash_attn2_forward_LookM
        from .qwen_model_croatt import (
            qwen_vl_forward_cross_attn,
            cross_attn_qwen_decoder_forward,
            cross_attn_qwen_forward_unflatten,
        )
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen_vl_forward_cross_attn
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = cross_attn_qwen_forward_unflatten
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = cross_attn_qwen_decoder_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_LookM

    elif method == "lookm_kvrepack":
        print("Using LookM-KVRepack!")
        from .qwen_model_lookm_kvrepack import qwen_flash_attn2_forward_LookMKVRepack
        from .qwen_model_croatt import (
            qwen_vl_forward_cross_attn,
            cross_attn_qwen_decoder_forward,
            cross_attn_qwen_forward_unflatten,
        )
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen_vl_forward_cross_attn
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = cross_attn_qwen_forward_unflatten
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = cross_attn_qwen_decoder_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_LookMKVRepack

    elif method == "madakv":
        print("Using MadaKV!")
        from .qwen_model import qwen_flash_attn2_forward_MadaKV
        from .qwen_model_croatt import (
            qwen_vl_forward_cross_attn,
            cross_attn_qwen_decoder_forward_MadaKV,
            cross_attn_qwen_forward_unflatten_MadaKV,
        )
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen_vl_forward_cross_attn
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = cross_attn_qwen_forward_unflatten_MadaKV
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = cross_attn_qwen_decoder_forward_MadaKV
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_MadaKV

    elif method == "madakv_kvrepack":
        print("Using MadaKV-KVRepack!")
        from .qwen_model_madakv_kvrepack import qwen_flash_attn2_forward_MadaKVKVRepack
        from .qwen_model_croatt import (
            qwen_vl_forward_cross_attn,
            cross_attn_qwen_decoder_forward_MadaKV,
            cross_attn_qwen_forward_unflatten_MadaKV,
        )
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel.forward = qwen_vl_forward_cross_attn
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = cross_attn_qwen_forward_unflatten_MadaKV
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer.forward = cross_attn_qwen_decoder_forward_MadaKV
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_MadaKVKVRepack

    elif method == "randomkv":
        print("Using RandomKV!")
        from .qwen_model_randomkv import qwen_flash_attn2_forward_RandomKV, randomkv_qwen_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = randomkv_qwen_forward
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_RandomKV

    elif method == "snapkv_kvrepack":
        print("Using SnapKV-KVRepack!")
        from .qwen_model_snapkv_kvrepack import qwen_flash_attn2_forward_SnapKVKVRepack
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLFlashAttention2.forward = qwen_flash_attn2_forward_SnapKVKVRepack

    else:
        raise NotImplementedError(f"Unsupported qwen method: {method}")

    if method not in ["fullkv"]:

        original_prepare_inputs = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation
        def prepare_inputs_for_generation_qwen(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            second_per_grid_ts=None,
            **kwargs,
        ):

            if past_key_values is not None and not isinstance(past_key_values, tuple):
                if len(past_key_values.key_cache) == 0:
                    for layer in self.model.language_model.layers:
                        layer.self_attn.kv_seq_len = 0

            return original_prepare_inputs(
                self,
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                use_cache=use_cache,
                **kwargs,
            )
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen


def replace_llava_onevision(method):
    from kvrepack.qwen2_model import (
        qwen2_flash_attn2_forward_SnapKV,
        qwen2_flash_attn2_forward_SparseMM,
        qwen2_flash_attn2_forward_LookM,
        qwen2_flash_attn2_forward_MadaKV,
        sparsemm_qwen2_forward,
        lookm_qwen2_forward,
        madakv_qwen2_forward,
        decoder_qwen2_forward_sparsemm,
        decoder_qwen2_forward_lookm,
        decoder_qwen2_forward_madakv,
    )
    from kvrepack.modeling_llava_onevision import LlavaOnevisionModel

    if method == "snapkv":
        print("Using SnapKV on LlavaOneVision!")
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_flash_attn2_forward_SnapKV
    
    elif method == "sparsemm":
        print("Using SparseMM on LlavaOneVision!")
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = sparsemm_qwen2_forward  # DynamicSplitCache
        transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer.forward = decoder_qwen2_forward_sparsemm # decoder need past the past_key_value
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_flash_attn2_forward_SparseMM
    
    elif method == "lookm":
        print("Using LookM on LlavaOneVision!")
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionModel.forward = LlavaOnevisionModel.forward # is_vision_token_mask past
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = lookm_qwen2_forward # past vision mask
        transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer.forward = decoder_qwen2_forward_lookm # past vision mask
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_flash_attn2_forward_LookM # past vision mask & actual pruning

    elif method == "madakv":
        print("Using MadaKV in LlavaOneVison!")
        transformers.models.llava_onevision.modeling_llava_onevision.LlavaOnevisionModel.forward = LlavaOnevisionModel.forward # is_vision_token_mask past
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = madakv_qwen2_forward # past vision mask & next-layer budget
        transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer.forward = decoder_qwen2_forward_madakv # past vision mask & next-layer budget
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_flash_attn2_forward_MadaKV # past vision mask & next-layer budget with updating
    else:
        raise NotImplementedError(f"Unsupported LlavaOnevision method: {method}")
