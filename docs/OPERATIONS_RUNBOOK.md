# BIZRA AEON OMEGA - Operations Runbook
# Elite Practitioner Grade | Production Operations Guide
# Version: 1.0.0 | Classification: OPERATIONAL

═══════════════════════════════════════════════════════════════════════════════
                    ب ز ر ع   |   BIZRA OPERATIONS RUNBOOK
                 "Excellence Through Vigilant Operations"
═══════════════════════════════════════════════════════════════════════════════

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Quick Reference Commands](#quick-reference-commands)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Scaling Operations](#scaling-operations)
6. [Incident Response](#incident-response)
7. [Backup & Recovery](#backup--recovery)
8. [Security Operations](#security-operations)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Maintenance Windows](#maintenance-windows)

---

## 1. System Overview

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BIZRA AEON OMEGA Production Stack                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Ingress    │────▶│  LoadBalancer│────▶│   API Pods   │                 │
│  │   (NGINX)    │     │   (Layer 4)  │     │   (12 min)   │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│         │                                        │                          │
│         ▼                                        ▼                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │  Prometheus  │◀────│   Metrics    │     │    Redis     │                 │
│  │   + Grafana  │     │   Exporter   │     │   Cluster    │                 │
│  └──────────────┘     └──────────────┘     └──────────────┘                 │
│                                                   │                          │
│                                                   ▼                          │
│                            ┌──────────────────────────────┐                 │
│                            │   Neo4j Graph Database       │                 │
│                            │   (3-node cluster)           │                 │
│                            └──────────────────────────────┘                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Metrics

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Response Time (p95) | < 100ms | > 500ms |
| Error Rate | < 0.1% | > 1% |
| CPU Utilization | < 60% | > 85% |
| Memory Usage | < 70% | > 90% |
| Pod Availability | > 99.9% | < 99% |
| FAISS Query Time | < 50ms | > 200ms |

---

## 2. Quick Reference Commands

### Kubernetes Status Commands

```bash
# Namespace overview
kubectl get all -n bizra-system

# Pod status with details
kubectl get pods -n bizra-system -o wide

# Pod logs (current)
kubectl logs -f deployment/bizra-api -n bizra-system

# Pod logs (previous crashed container)
kubectl logs deployment/bizra-api -n bizra-system --previous

# HPA status
kubectl get hpa -n bizra-system

# Resource usage
kubectl top pods -n bizra-system
kubectl top nodes

# Describe deployment
kubectl describe deployment bizra-api -n bizra-system

# Get events (troubleshooting)
kubectl get events -n bizra-system --sort-by='.lastTimestamp'
```

### Health Checks

```bash
# API health endpoint
curl -s https://api.bizra.io/health | jq

# Readiness probe
curl -s https://api.bizra.io/ready | jq

# Metrics endpoint
curl -s https://api.bizra.io/metrics

# Ihsān scoring test
curl -X POST https://api.bizra.io/score \
  -H "Content-Type: application/json" \
  -d '{"content": "test attestation"}'
```

---

## 3. Deployment Procedures

### Standard Deployment (GitHub Actions)

1. **Merge to main branch** → Triggers CI/CD pipeline
2. **Monitor Actions** → Check GitHub Actions tab
3. **Verify staging** → Test at staging.bizra.io
4. **Approve production** → Manual approval step
5. **Monitor rollout** → Watch deployment status

### Manual Deployment

```bash
# 1. Build and push image
docker build -t ghcr.io/bizra/bizra-omega:v1.x.x -f docker/Dockerfile .
docker push ghcr.io/bizra/bizra-omega:v1.x.x

# 2. Update deployment image
kubectl set image deployment/bizra-api \
  bizra-api=ghcr.io/bizra/bizra-omega:v1.x.x \
  -n bizra-system

# 3. Monitor rollout
kubectl rollout status deployment/bizra-api -n bizra-system

# 4. Verify health
kubectl get pods -n bizra-system -l app=bizra-api
```

### Rollback Procedure

```bash
# View rollout history
kubectl rollout history deployment/bizra-api -n bizra-system

# Rollback to previous version
kubectl rollout undo deployment/bizra-api -n bizra-system

# Rollback to specific revision
kubectl rollout undo deployment/bizra-api -n bizra-system --to-revision=3

# Verify rollback
kubectl rollout status deployment/bizra-api -n bizra-system
```

### Blue-Green Deployment

```bash
# Create green deployment
kubectl apply -f k8s/deployment-green.yaml

# Test green deployment
kubectl port-forward svc/bizra-api-green 8081:80 -n bizra-system

# Switch traffic to green
kubectl patch service bizra-api -n bizra-system \
  -p '{"spec":{"selector":{"version":"green"}}}'

# Cleanup blue after verification
kubectl delete deployment bizra-api-blue -n bizra-system
```

---

## 4. Monitoring & Alerting

### Prometheus Queries

```promql
# Request rate
rate(bizra_requests_total[5m])

# Error rate percentage
100 * rate(bizra_requests_total{status=~"5.."}[5m]) / rate(bizra_requests_total[5m])

# Average response time
histogram_quantile(0.95, rate(bizra_request_duration_seconds_bucket[5m]))

# Memory usage percentage
100 * container_memory_usage_bytes{namespace="bizra-system"} / container_spec_memory_limit_bytes

# CPU utilization
100 * rate(container_cpu_usage_seconds_total{namespace="bizra-system"}[5m])

# FAISS index size
bizra_faiss_index_size

# Active attestations
bizra_active_attestations_total
```

### Grafana Dashboards

| Dashboard | URL | Purpose |
|-----------|-----|---------|
| BIZRA Overview | /d/bizra-main | System-wide health |
| API Performance | /d/bizra-api | Request/response metrics |
| Ihsān Scoring | /d/bizra-ihsan | Scoring dimension analysis |
| Infrastructure | /d/bizra-infra | K8s/resource metrics |

### Alert Configuration

```yaml
# Prometheus AlertManager rules
groups:
  - name: bizra-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(bizra_requests_total{status=~"5.."}[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(bizra_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
          
      - alert: PodNotReady
        expr: kube_pod_status_ready{namespace="bizra-system"} == 0
        for: 5m
        labels:
          severity: critical
```

---

## 5. Scaling Operations

### Horizontal Pod Autoscaler

```bash
# View current HPA status
kubectl get hpa bizra-api-hpa -n bizra-system

# View HPA details
kubectl describe hpa bizra-api-hpa -n bizra-system

# Manual scale override
kubectl scale deployment bizra-api -n bizra-system --replicas=20

# Reset to HPA control
kubectl annotate deployment bizra-api -n bizra-system \
  kubernetes.io/change-cause="Reset to HPA control"
```

### Scaling Thresholds

| Metric | Scale Up | Scale Down | Cooldown |
|--------|----------|------------|----------|
| CPU | > 70% | < 40% | 300s |
| Memory | > 80% | < 50% | 300s |
| Request Rate | > 1000/s | < 500/s | 300s |

### Vertical Pod Autoscaler

```bash
# View VPA recommendations
kubectl describe vpa bizra-api-vpa -n bizra-system

# Apply VPA recommendations manually
kubectl patch deployment bizra-api -n bizra-system \
  -p '{"spec":{"template":{"spec":{"containers":[{
    "name":"bizra-api",
    "resources":{"requests":{"cpu":"500m","memory":"1Gi"}}
  }]}}}}'
```

---

## 6. Incident Response

### Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| SEV1 | System down | < 15 min | Immediate page |
| SEV2 | Degraded service | < 30 min | On-call team |
| SEV3 | Minor issues | < 2 hours | Normal process |
| SEV4 | Cosmetic | Next business day | Ticket |

### Incident Response Checklist

#### SEV1 - System Down

```markdown
□ 1. Acknowledge alert
□ 2. Start incident channel (#incident-YYYYMMDD)
□ 3. Check pod status: kubectl get pods -n bizra-system
□ 4. Check events: kubectl get events -n bizra-system --sort-by='.lastTimestamp'
□ 5. Check logs: kubectl logs -l app=bizra-api -n bizra-system --tail=100
□ 6. Attempt rollback if deployment-related
□ 7. Scale up if capacity-related
□ 8. Check dependencies (Neo4j, Redis)
□ 9. Update status page
□ 10. Post-incident review within 24 hours
```

#### Common Issue Quick Fixes

**Pods in CrashLoopBackOff:**
```bash
# Get pod details
kubectl describe pod <pod-name> -n bizra-system

# Check previous logs
kubectl logs <pod-name> -n bizra-system --previous

# Delete and let deployment recreate
kubectl delete pod <pod-name> -n bizra-system
```

**OOMKilled:**
```bash
# Increase memory limits
kubectl patch deployment bizra-api -n bizra-system \
  -p '{"spec":{"template":{"spec":{"containers":[{
    "name":"bizra-api",
    "resources":{"limits":{"memory":"4Gi"}}
  }]}}}}'
```

**ImagePullBackOff:**
```bash
# Check image pull secrets
kubectl get secrets -n bizra-system

# Verify registry credentials
kubectl get secret ghcr-credentials -n bizra-system -o yaml
```

---

## 7. Backup & Recovery

### Backup Schedule

| Component | Frequency | Retention | Location |
|-----------|-----------|-----------|----------|
| Neo4j | Hourly | 7 days | S3/Azure Blob |
| Redis | Every 5 min | 24 hours | S3/Azure Blob |
| FAISS Indexes | Daily | 30 days | S3/Azure Blob |
| Configs | On change | 90 days | Git |

### Neo4j Backup

```bash
# Trigger backup
kubectl exec -it neo4j-0 -n bizra-system -- \
  neo4j-admin backup --backup-dir=/backup --database=neo4j

# Verify backup
kubectl exec -it neo4j-0 -n bizra-system -- ls -la /backup

# Copy backup to local
kubectl cp bizra-system/neo4j-0:/backup ./neo4j-backup
```

### Recovery Procedures

**Neo4j Recovery:**
```bash
# Stop the database
kubectl scale statefulset neo4j -n bizra-system --replicas=0

# Copy backup to pod
kubectl cp ./neo4j-backup bizra-system/neo4j-0:/backup

# Restore
kubectl exec -it neo4j-0 -n bizra-system -- \
  neo4j-admin restore --from=/backup --database=neo4j --force

# Start database
kubectl scale statefulset neo4j -n bizra-system --replicas=3
```

**FAISS Index Recovery:**
```bash
# Download latest backup
az storage blob download \
  --container-name backups \
  --name faiss/latest.index \
  --file faiss-recovery.index

# Copy to pods
for pod in $(kubectl get pods -n bizra-system -l app=bizra-api -o name); do
  kubectl cp faiss-recovery.index bizra-system/${pod#pod/}:/data/faiss/main.index
done

# Restart pods to load index
kubectl rollout restart deployment/bizra-api -n bizra-system
```

---

## 8. Security Operations

### Certificate Management

```bash
# Check certificate expiry
kubectl get certificate -n bizra-system

# Force certificate renewal
kubectl delete certificate bizra-tls -n bizra-system
kubectl apply -f k8s/ingress.yaml

# View certificate details
kubectl describe certificate bizra-tls -n bizra-system
```

### Secret Rotation

```bash
# Rotate Neo4j password
kubectl create secret generic neo4j-credentials \
  -n bizra-system \
  --from-literal=NEO4J_PASSWORD='<new-password>' \
  --dry-run=client -o yaml | kubectl apply -f -

# Rotate API keys
kubectl create secret generic api-secrets \
  -n bizra-system \
  --from-literal=SECRET_KEY='<new-key>' \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods to pick up new secrets
kubectl rollout restart deployment/bizra-api -n bizra-system
```

### Security Audit Commands

```bash
# List pods with elevated privileges
kubectl get pods -n bizra-system -o json | jq '.items[] | select(.spec.securityContext.privileged==true)'

# Check network policies
kubectl get networkpolicies -n bizra-system

# Verify RBAC
kubectl auth can-i --list --as=system:serviceaccount:bizra-system:bizra-api
```

---

## 9. Troubleshooting Guide

### Common Issues

#### Issue: High Memory Usage

**Symptoms:** Pods OOMKilled, slow responses
**Diagnosis:**
```bash
kubectl top pods -n bizra-system
kubectl describe pod <pod-name> -n bizra-system | grep -A5 "Last State"
```
**Resolution:**
1. Check for memory leaks in application logs
2. Increase memory limits
3. Scale horizontally instead of vertically
4. Review FAISS index size

#### Issue: Slow FAISS Queries

**Symptoms:** p95 latency > 200ms for search operations
**Diagnosis:**
```bash
# Check index size
curl -s https://api.bizra.io/metrics | grep faiss

# Check vector count
kubectl exec -it deployment/bizra-api -n bizra-system -- \
  python -c "import faiss; idx = faiss.read_index('/data/faiss/main.index'); print(idx.ntotal)"
```
**Resolution:**
1. Rebuild index with IVF partitioning
2. Reduce nprobe parameter
3. Add more replicas for query distribution
4. Consider GPU-accelerated FAISS

#### Issue: Neo4j Connection Failures

**Symptoms:** Graph queries timeout, connection refused errors
**Diagnosis:**
```bash
kubectl logs neo4j-0 -n bizra-system --tail=50
kubectl exec -it neo4j-0 -n bizra-system -- cypher-shell "CALL dbms.cluster.overview()"
```
**Resolution:**
1. Check cluster health
2. Verify network policies allow traffic
3. Restart Neo4j pods one at a time
4. Check disk space

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/bizra-api -n bizra-system LOG_LEVEL=DEBUG

# Exec into pod for debugging
kubectl exec -it deployment/bizra-api -n bizra-system -- bash

# Run diagnostics
python -c "
from cognitive_sovereign import QuantumTemporalSecurity, IhsanPrinciples
qts = QuantumTemporalSecurity()
print(f'Ihsān Weights: {IhsanPrinciples.default_weights()}')
print(f'Security Status: OK')
"

# Disable debug logging
kubectl set env deployment/bizra-api -n bizra-system LOG_LEVEL=INFO
```

---

## 10. Maintenance Windows

### Scheduled Maintenance

| Day | Time (UTC) | Duration | Purpose |
|-----|------------|----------|---------|
| Sunday | 02:00 | 2 hours | Updates/Patches |
| 1st of month | 04:00 | 4 hours | Major upgrades |
| Quarterly | 06:00 | 8 hours | Security audit |

### Maintenance Mode

```bash
# Enable maintenance mode
kubectl scale deployment bizra-api -n bizra-system --replicas=3
kubectl patch ingress bizra-ingress -n bizra-system \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/custom-http-errors":"503"}}}'

# Display maintenance page
kubectl apply -f k8s/maintenance-page.yaml

# Perform maintenance...

# Disable maintenance mode
kubectl delete -f k8s/maintenance-page.yaml
kubectl scale deployment bizra-api -n bizra-system --replicas=12
kubectl patch ingress bizra-ingress -n bizra-system \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/custom-http-errors":""}}}'
```

---

## 📞 Contact Information

| Role | Contact | Availability |
|------|---------|--------------|
| On-Call Engineer | pager@bizra.io | 24/7 |
| Platform Lead | platform@bizra.io | Business hours |
| Security Team | security@bizra.io | 24/7 for SEV1 |
| Database Admin | dba@bizra.io | Business hours |

---

## 📝 Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-01-26 | BIZRA Team | Initial release |

---

**إِنَّا لِلَّهِ وَإِنَّا إِلَيْهِ رَاجِعُونَ**  
*"Indeed, we belong to Allah, and indeed to Him we return."*

*Operations excellence through vigilance, humility, and continuous improvement.*
