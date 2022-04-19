# sentence-transformers-huggingface-inferentia

In this end-to-end tutorial, you will learn how to speed up [Sentence-Transformers](https://www.sbert.net/) like  SBERT for creating sentence embedding using Hugging Face Transformers, Amazon SageMaker, and AWS Inferentia. 

You will learn how to: 

1. Convert your Sentence-Transformer to AWS Neuron (Inferentia)
2. Create a custom `inference.py` script for `sentence-embeddings`
3. Create and upload the neuron model and inference script to Amazon S3
4. Deploy a Real-time Inference Endpoint on Amazon SageMaker
5. Run and evaluate Inference performance of BERT on Inferentia

Let's get started! 🚀

---

*If you are going to use Sagemaker in a local environment (not SageMaker Studio or Notebook Instances). You need access to an IAM Role with the required permissions for Sagemaker. You can find [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) more about it.*