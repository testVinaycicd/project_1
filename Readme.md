export PIPELINE_VERSION=2.14.3

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"

kubectl wait --for=condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${PIPELINE_VERSION}"

kubectl set image deployment/minio minio=minio/minio:RELEASE.2023-03-20T20-16-18Z -n kubeflow






kubectl edit configmap workflow-controller-configmap -n kubeflow





data:
    workflowDefaults: |
        spec:
            securityContext:
                runAsNonRoot: false

kubectl rollout restart deployment workflow-controller -n kubeflow

kubectl delete workflow -n kubeflow --all