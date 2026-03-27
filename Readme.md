kubectl set image deployment/minio -n kubeflow minio=bitnami/minio:latest




kubectl edit configmap workflow-controller-configmap -n kubeflow





data:
    workflowDefaults: |
        spec:
            securityContext:
                runAsNonRoot: false

kubectl rollout restart deployment workflow-controller -n kubeflow

kubectl delete workflow -n kubeflow --all