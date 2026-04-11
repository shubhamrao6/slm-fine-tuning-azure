# Azure Resource Cost Estimate

Subscription: Ether_POC_Subscription (fe37b5f6-efa5-43a5-ba04-2d3684b07345)
User: shubham@cashapi.in
Date: April 8, 2026

---

## 1. Virtual Machines

| Resource | Size | Location | Status | Est. Monthly Cost |
|----------|------|----------|--------|-------------------|
| chaosai | Standard_D2s_v3 (2 vCPU, 8 GB RAM) | East US | **Running** | ~$70/mo |

> VM is running 24/7. At ~$0.096/hr, that's roughly $70/mo. Deallocating when not in use would bring this to $0.

---

## 2. Managed Disks

| Resource | Size | SKU | State | Est. Monthly Cost |
|----------|------|-----|-------|-------------------|
| chaosai_OsDisk (VM OS disk) | 40 GB | Premium_LRS (P6 tier) | Attached | ~$10/mo |

> Premium SSD P6 (64 GB tier) is billed even when VM is deallocated.

---

## 3. Cosmos DB Accounts

| Resource | Kind | Mode | Location | Est. Monthly Cost |
|----------|------|------|----------|-------------------|
| sqlcosmossanta | NoSQL | Serverless | Central India | ~$0–5/mo (usage-based) |
| cashapi-cosmosdb | NoSQL | Serverless | Central India | ~$0–5/mo (usage-based) |
| demoacccosmo123 | Gremlin (Graph) | Provisioned | West US | ~$24+/mo (min 400 RU/s) |

> Serverless accounts charge per RU consumed ($0.25 per 1M RUs) + $0.25/GB storage. Minimal usage = near zero.
> The Gremlin (provisioned) account has a minimum of 400 RU/s at ~$0.008/hr per 100 RU/s = ~$24/mo baseline even if idle.

---

## 4. Cognitive Services / AI

| Resource | Kind | SKU | Location | Est. Monthly Cost |
|----------|------|-----|----------|-------------------|
| demorao | Custom Vision Training | F0 (Free) | East US | $0 |
| demorao-Prediction | Custom Vision Prediction | F0 (Free) | East US | $0 |
| DocumentRecognizerOS | Form Recognizer | F0 (Free) | East US | $0 |
| ReadifyAI-mvp | AI Services | S0 | East US | Pay-per-use |
| ether-openai | OpenAI | S0 | East US 2 | Pay-per-use |
| ether-project-resource | AI Services | S0 | East US 2 | Pay-per-use |

> F0 SKUs are free tier with limited quotas. S0 resources are pay-per-use (charged per API call / token). If idle, cost is $0. Active usage of OpenAI models can vary widely ($1–$100+/mo depending on volume).

---

## 5. App Service Plans

| Plan | SKU | Tier | Location | Est. Monthly Cost |
|------|-----|------|----------|-------------------|
| ASP-PMService-97fe | F1 | Free | Central US | $0 |
| ASP-demo-8328 | Y1 | Dynamic (Consumption) | Central US | ~$0 (pay-per-execution) |
| ASP-demo-92d8 | Y1 | Dynamic | Central US | ~$0 |
| ASP-demo-bfa0 | F1 | Free | Central India | $0 |
| test-app-nihal | F1 | Free | Central India | $0 |
| ASP-CashAPI-9e04 | Y1 | Dynamic | Central India | ~$0 |
| ASP-CashAPI-b64c | Y1 | Dynamic | Central India | ~$0 |
| ASP-CashAPI-9e6f | Y1 | Dynamic | Central India | ~$0 |
| ASP-Ecosustain-8847 | Y1 | Dynamic | Central India | ~$0 |
| ASP-DateGain-a5da | Y1 | Dynamic | Central India | ~$0 |
| ASP-demo-8bd6 | F1 | Free | UK South | $0 |
| ASP-demo-b49a | F1 | Free | UK West | $0 |
| ASP-CashAPI-86cf | F1 | Free | South India | $0 |
| ASP-NihalExploration-921b | F1 | Free | East Asia | $0 |

> All plans are either Free (F1) or Consumption/Dynamic (Y1). No paid App Service Plans. Cost is effectively $0 unless consumption functions are heavily invoked.

---

## 6. Static Web Apps

| Resource | SKU | Location | Est. Monthly Cost |
|----------|-----|----------|-------------------|
| cash-api-frontend | Free | West US 2 | $0 |
| EtherPOC | Free | Central US | $0 |
| ReadifyAI | Free | Central US | $0 |
| dfmeatechnosol | Free | East Asia | $0 |
| dfmea-technosol | Free | East Asia | $0 |

> All on Free tier.

---

## 7. API Management

| Resource | SKU | Location | Est. Monthly Cost |
|----------|-----|----------|-------------------|
| cashapiconsumption | Consumption | Central India | ~$3.50/mo per million calls |

> Consumption tier: first 1M calls/mo free, then $3.50 per million. If low traffic, effectively near $0.

---

## 8. Storage Accounts

| Resource | SKU | Kind | Location | Est. Monthly Cost |
|----------|-----|------|----------|-------------------|
| lumostore5344 | Standard_RAGRS | StorageV2 | East US | ~$1–5/mo |
| raokaworkspace5618175407 | Standard_LRS | StorageV2 | East US | ~$0.50–2/mo |
| cashapifunctions | Standard_LRS | Storage | Central India | ~$0.50–2/mo |

> Storage costs depend on data volume. At low usage (< 10 GB), these are minimal. RA-GRS is more expensive due to geo-redundant replication.

---

## 9. Communication Services

| Resource | Location | Est. Monthly Cost |
|----------|----------|-------------------|
| cashapi-communication | Global | Pay-per-use |
| cashapi-email (domain: cashapi.in) | Global | Pay-per-use |
| vner-communication | Global | Pay-per-use |
| vner-email-notifications (domain: vner.in) | Global | Pay-per-use |

> Email: ~$0.00025 per email sent. If idle, $0.

---

## 10. Other Resources (Low/No Cost)

| Category | Resources | Est. Monthly Cost |
|----------|-----------|-------------------|
| Network Watchers | 2 (eastus, centralus) | $0 (free) |
| NSGs | 3 (ThinkingPad, DUB-CL1, chaosai) | $0 (free) |
| Public IPs | 2 (ThinkingPad-ip, chaosai-ip) | ~$4/mo each if static |
| VNets | 3 (demo-vnet, vnet-eastus, DUB-CL1-vnet) | $0 (free, peering costs extra) |
| Network Interfaces | 2 | $0 |
| Data Share Accounts | 6 | $0 (pay per snapshot) |
| Data Factory | 1 (demofactorydata11220) | Pay-per-activity |
| Logic Apps | 1 (Get-GeoFromIpAndTagIncident) | ~$0.000025/action |
| Application Insights | 7 components | Free up to 5 GB/mo ingestion |
| Log Analytics Workspaces | 4 | Free up to 5 GB/mo ingestion |
| Key Vault | 1 (raokaworkspace8759957443) | ~$0.03/10K operations |
| Event Grid System Topic | 1 | $0.60 per million operations |
| Azure AD B2C | 1 (cashapiprototype) | First 50K auth free/mo |
| Bot Service | 1 (purview-mcp45192) | Free tier likely |
| Smart Detection Alert Rules | 7 | $0 (free) |
| Data Collection Rules/Endpoints | 3 | $0 (ingestion-based) |
| Power BI Private Link | 1 | $0 (part of Power BI licensing) |

---

## Summary: Estimated Monthly Cost

| Category | Est. Monthly Cost |
|----------|-------------------|
| VM (chaosai - running 24/7) | **~$70** |
| Managed Disk (Premium SSD 40GB) | **~$10** |
| Cosmos DB (Gremlin provisioned) | **~$24** |
| Cosmos DB (2x Serverless) | **~$0–10** |
| Public IPs (2x static) | **~$8** |
| Storage Accounts (3x) | **~$2–9** |
| Cognitive Services (S0, usage-based) | **$0–100+** (depends on usage) |
| Everything else (Free/Consumption tier) | **~$0–5** |
| | |
| **Total (idle/low usage)** | **~$115–135/mo** |
| **Total (moderate AI/API usage)** | **~$150–250/mo** |

---

## Top Cost Saving Recommendations

1. **Deallocate the `chaosai` VM** when not in use — saves ~$70/mo instantly
2. **Review `demoacccosmo123`** (Gremlin/provisioned) — if unused, delete or switch to serverless to save ~$24/mo
3. **Check OpenAI/AI Services usage** — S0 resources are pay-per-use but can add up quickly with active workloads
4. **Release unused Public IPs** — static IPs cost ~$4/mo each even when unattached

> Pricing sourced from [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) and [Holori Azure VM pricing](https://calculator.holori.com/azure/vm/standard-d2s-v3). Actual costs may vary based on usage, region, and any applicable discounts. Content was rephrased for compliance with licensing restrictions.
