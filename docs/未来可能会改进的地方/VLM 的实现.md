`llm_emv\vlm.py` 里的：

```python
@property
    def model(self) -> BaseChatModel:
        raise NotImplementedError

    def prepare_multimodal_message_content(self, *args: Union[str, Image]) -> List[dict]:
        raise NotImplementedError
```

都还没实现


