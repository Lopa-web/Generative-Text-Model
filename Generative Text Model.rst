.. code:: ipython3

    !pip install transformers torch


.. parsed-literal::

    Requirement already satisfied: transformers in c:\users\kiit0001\anaconda3\lib\site-packages (4.32.1)
    Requirement already satisfied: torch in c:\users\kiit0001\anaconda3\lib\site-packages (2.6.0)
    Requirement already satisfied: filelock in c:\users\kiit0001\anaconda3\lib\site-packages (from transformers) (3.9.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.15.1 in c:\users\kiit0001\anaconda3\lib\site-packages (from transformers) (0.24.6)
    Requirement already satisfied: numpy>=1.17 in c:\users\kiit0001\anaconda3\lib\site-packages (from transformers) (2.0.2)
    Requirement already satisfied: packaging>=20.0 in c:\users\kiit0001\anaconda3\lib\site-packages (from transformers) (23.1)
    Requirement already satisfied: pyyaml>=5.1 in c:\users\kiit0001\anaconda3\lib\site-packages (from transformers) (6.0.2)
    Requirement already satisfied: regex!=2019.12.17 in c:\users\kiit0001\anaconda3\lib\site-packages (from transformers) (2022.7.9)
    Requirement already satisfied: requests in c:\users\kiit0001\anaconda3\lib\site-packages (from transformers) (2.32.3)
    Requirement already satisfied: tokenizers!=0.11.3,<0.14,>=0.11.1 in c:\users\kiit0001\anaconda3\lib\site-packages (from transformers) (0.13.2)
    Requirement already satisfied: safetensors>=0.3.1 in c:\users\kiit0001\anaconda3\lib\site-packages (from transformers) (0.3.2)
    Requirement already satisfied: tqdm>=4.27 in c:\users\kiit0001\anaconda3\lib\site-packages (from transformers) (4.65.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in c:\users\kiit0001\anaconda3\lib\site-packages (from torch) (4.12.2)
    Requirement already satisfied: networkx in c:\users\kiit0001\anaconda3\lib\site-packages (from torch) (3.1)
    Requirement already satisfied: jinja2 in c:\users\kiit0001\anaconda3\lib\site-packages (from torch) (3.1.2)
    Requirement already satisfied: fsspec in c:\users\kiit0001\anaconda3\lib\site-packages (from torch) (2024.12.0)
    Requirement already satisfied: sympy==1.13.1 in c:\users\kiit0001\anaconda3\lib\site-packages (from torch) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\kiit0001\anaconda3\lib\site-packages (from sympy==1.13.1->torch) (1.3.0)
    Requirement already satisfied: colorama in c:\users\kiit0001\anaconda3\lib\site-packages (from tqdm>=4.27->transformers) (0.4.6)
    Requirement already satisfied: MarkupSafe>=2.0 in c:\users\kiit0001\anaconda3\lib\site-packages (from jinja2->torch) (2.1.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\kiit0001\anaconda3\lib\site-packages (from requests->transformers) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\kiit0001\anaconda3\lib\site-packages (from requests->transformers) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\kiit0001\anaconda3\lib\site-packages (from requests->transformers) (1.26.16)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\kiit0001\anaconda3\lib\site-packages (from requests->transformers) (2024.12.14)
    

.. code:: ipython3

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import torch


.. parsed-literal::

    C:\Users\KIIT0001\anaconda3\Lib\site-packages\transformers\utils\generic.py:260: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
      torch.utils._pytree._register_pytree_node(
    

.. code:: ipython3

    model_name = "gpt2"  
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model.eval()


.. parsed-literal::

    C:\Users\KIIT0001\anaconda3\Lib\site-packages\huggingface_hub\file_download.py:1150: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(
    



.. parsed-literal::

    GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0-11): 12 x GPT2Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): GPT2Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): GPT2MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (act): NewGELUActivation()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )



.. code:: ipython3

    def generate_text(prompt, max_length=200):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7, top_p=0.9, top_k=50)
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

.. code:: ipython3

    user_prompt = input("Enter a prompt: ")


.. parsed-literal::

    Enter a prompt:  "The future of artificial intelligence is"
    

.. code:: ipython3

    generated_paragraph = generate_text(user_prompt, max_length=250)
    
    print("\nGenerated Paragraph based on your prompt:")
    print(generated_paragraph)


.. parsed-literal::

    C:\Users\KIIT0001\anaconda3\Lib\site-packages\transformers\generation\configuration_utils.py:362: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.7` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
      warnings.warn(
    C:\Users\KIIT0001\anaconda3\Lib\site-packages\transformers\generation\configuration_utils.py:367: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
      warnings.warn(
    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    

.. parsed-literal::

    
    Generated Paragraph based on your prompt:
    "The future of artificial intelligence is"
    
    "We are not going to be able to predict the future, but we are going be very confident that we will be in the next few years."
    .
    , "The Future of Artificial Intelligence is a book that will help you understand the challenges of AI and how we can help solve them."
    

