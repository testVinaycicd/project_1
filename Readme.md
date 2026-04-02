export PIPELINE_VERSION=2.14.3

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=${PIPELINE_VERSION}"

kubectl wait --for=condition=established --timeout=60s crd/applications.app.k8s.io

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=${PIPELINE_VERSION}"

kubectl set image deployment/minio minio=minio/minio:RELEASE.2023-03-20T20-16-18Z -n kubeflow

kubectl apply -f congig.yaml




kubectl edit configmap workflow-controller-configmap -n kubeflow





data:
    workflowDefaults: |
      spec:
        securityContext:
          runAsNonRoot: false

kubectl rollout restart deployment workflow-controller -n kubeflow

kubectl delete workflow -n kubeflow --all


Postgress setting up mlflow db

-- 1. Create database
CREATE DATABASE mlflow;

-- 2. Create user
CREATE USER mlflow_user WITH PASSWORD 'mlflow_password';

-- 3. Transfer database ownership
ALTER DATABASE mlflow OWNER TO mlflow_user;

-- 4. Connect to the database
\c mlflow

-- 5. Transfer schema ownership
ALTER SCHEMA public OWNER TO mlflow_user;

-- 6. Ensure required privileges (explicit, avoids edge cases)
GRANT USAGE, CREATE ON SCHEMA public TO mlflow_user;

-- 7. Default privileges for future objects (important for migrations)
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT ALL ON TABLES TO mlflow_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT ALL ON SEQUENCES TO mlflow_user;



helm install mlflow community-charts/mlflow \
--namespace mlflow \
--set backendStore.databaseMigration=true \
--set backendStore.postgres.enabled=true \
--set backendStore.postgres.host=your-postgres-host \
--set backendStore.postgres.port=5432 \
--set backendStore.postgres.database=mlflow \
--set backendStore.postgres.user=mlflow_user \
--set backendStore.postgres.password=your_secure_password