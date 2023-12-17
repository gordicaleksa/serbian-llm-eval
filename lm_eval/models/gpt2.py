import torch
import transformers
from typing import Optional, Union
from lm_eval.base import BaseLM


def _get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HFLM(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        device="cuda:1",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
    ):
        super().__init__()

        # Initialize model
        if isinstance(pretrained, transformers.PreTrainedModel):
            self.model = pretrained
            self._device = self.model.device

            if tokenizer:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
            else:
                # Get tokenizer
                model_name = self.model.name_or_path
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                )

        elif isinstance(pretrained, str):

            # Initialize device
            assert isinstance(device, str)
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                print(f"Using device '{device}'")
            else:
                print("Device not specified")
                print(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            revision = revision + ("/" + subfolder if subfolder is not None else "")
            # pretrained = "jerteh/gpt2-orao"
            # Initialize new model and tokenizer instances
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                load_in_8bit=load_in_8bit,
                low_cpu_mem_usage=low_cpu_mem_usage,
                revision=revision,
                torch_dtype=_get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            ).to(self.device)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                tokenizer if tokenizer else pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
            # model_config = transformers.MistralConfig(**{  "_name_or_path": "/hdd/BalkanGPT/yugoGPT/checkpoint-18000_312/",  "architectures": [    "MistralForCausalLM"  ],  "bos_token_id": 1,  "eos_token_id": 2,  "hidden_act": "silu",  "hidden_size": 4096,  "initializer_range": 0.02,  "intermediate_size": 14336,  "max_position_embeddings": 32768,  "model_type": "mistral",  "num_attention_heads": 32,  "num_hidden_layers": 32,  "num_key_value_heads": 8,  "rms_norm_eps": 1e-05,  "rope_theta": 10000.0,  "sliding_window": 4096,  "tie_word_embeddings": False, "torch_dtype": "bfloat16", "transformers_version": "4.35.1", "use_cache": False, "vocab_size": 32000})
            # model_kwargs = {'device_map': None, 'torch_dtype': torch.bfloat16, 'use_flash_attention_2': True}
            # # Initialize new model and tokenizer instances
            # self.model = transformers.MistralForCausalLM.from_pretrained(
            #     pretrained,
            #     config=model_config,
            #     load_in_8bit=load_in_8bit,
            #     load_in_4bit=False,
            #     trust_remote_code=trust_remote_code,
            #     **model_kwargs,
            # )
            # self.model.tie_weights()

            # # for name, module in model.named_modules():
            # #     if "norm" in name:
            # #         module.to(torch.float32)
            # #     if model_config.model_type == "btlm":
            # #         # don't upcast lm_head for btlm
            # #         continue
            # #     if "lm_head" in name or "embed_tokens" in name:
            # #         if hasattr(module, "weight"):
            # #             module.to(torch.float32)
            # #         model.config.use_cache = False
            # #             model = model.to(cfg.device)
            # self.model.to(torch.bfloat16)
            # self.model.to(self.device)
            # self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            #     tokenizer if tokenizer else pretrained,
            #     # revision=revision,
            #     use_fast=True,
            #     trust_remote_code=trust_remote_code,
            # )

        else:
            raise TypeError(
                "Parameter pretrained should be of type str or transformers.PreTrainedModel"
            )

        self.model.eval()

        self.vocab_size = self.tokenizer.vocab_size

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs[
                "pad_token_id"
            ] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)


# for backwards compatibility
GPT2LM = HFLM

class OPTIMUMLM(BaseLM):
    def __init__(
        self,
        device="cpu",
        pretrained="gpt2",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        load_in_8bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
    ):
        super().__init__()

        import optimum
        from optimum.intel.openvino import OVModelForCausalLM

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int,str))

        device_list = set(["cuda", "cpu"] + [f'cuda:{i}' for i in range(torch.cuda.device_count())])
        if device and device in device_list:
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.gpt2 = OVModelForCausalLM.from_pretrained(
            pretrained,
            load_in_8bit=load_in_8bit,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_cache=True,
        )

        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained if tokenizer is None else tokenizer,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        except:
            print("Tokenizer is missed. Plaase save it into the same folder with the model.")

        self.vocab_size = self.tokenizer.vocab_size

        # setup for automatic batch size detection
        if batch_size == 'auto': 
            self.batch_size_per_gpu = batch_size
        else:
            self.batch_size_per_gpu = int(batch_size) 

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        return self.gpt2(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {'do_sample': False, 'max_length': max_length}
        if eos_token_id is not None:
            generation_kwargs['eos_token_id'] = eos_token_id
            generation_kwargs['pad_token_id'] = eos_token_id # setting eos_token_id as pad token
        return self.gpt2.generate(context, **generation_kwargs)
