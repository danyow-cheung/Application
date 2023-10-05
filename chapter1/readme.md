# Effective Planning of Deep Learning-Driven Projects

## Overview of DL projects 
1. project planning 
   總體而言，專案規劃應產生一份稱為行動手冊的文檔，該文檔包括如何實施和評估項目的全面描述。

2. building **minmum viable product(MVP)**
   展示專案價值的目標可交付成果的簡單版本。
   
   此階段的另一個重要面向是了解專案的困難，並遵循「快速失敗、經常失敗」的理念，拒絕風險較大或結果較差的路徑


3. building **fully featured product(FFP)**
   此階段的目標是完善 MVP，透過各種最佳化建置可投入生產的可交付成果。

4. Deployment and maintenance
   在許多情況下，部署設定與開發設定不同。 所以，將 FFP 投入生產時通常會涉及不同的工具集。

    此外，部署可能會引入開發過程中不可見的問題，這些問題主要是由於計算資源有限而出現的。
    因此，許多工程師和科學家在此階段花了額外的時間來改善使用者體驗。 大多數人認為部署是最後一步。


    然而，還有一步：維護。 需要持續監控資料品質和模型效能，為目標使用者提供穩定的服務
5. Project evaluation
   planning to evaluate whether the project has been carried out successfully or not. 


### Planning a DL project 
此階段應產生一份記錄良好的專案手冊，精確定義業務目標以及如何評估專案。 典型的劇本包含關鍵可交付成果的概述、利害關係人清單、定義步驟和瓶頸的甘特圖、職責定義、時間表和評估標準

#### Defining goal and evaluation metrics 
對於深度學習項目，有兩種類型的評估指標：**業務相關指標和基於模型的指標**。 與業務相關的指標的一些示例如下：轉換率、點擊率 (CTR)、生命週期價值、用戶參與度衡量標準、營運成本節省、投資回報率 (ROI) 和收入。 這些通常用於廣告、行銷和產品推薦垂直領域。

另一方面，基於模型的指標包括準確度、精確度、回想率、F1 分數、排名準確度指標、平均絕對誤差 (MAE)、均方誤差 (MSE)、均方根誤差 (RMSE) 和標準化平均絕對誤差（NMAE）。 一般來說，可以在各種指標之間進行權衡。 例如，如果滿足延遲要求對專案更為重要，則準確性的輕微降低可能是可以接受的。

#### Stakeholder identification
> 利害關係人識別

利害關係人可以分為兩類：內部利害關係人和外部利害關係人。 內部利害關係人是直接參與專案執行的利害關係人，而外部利害關係人可能處於圈子之外，以間接的方式支持專案執行
**Internal stakeholderes**
- Sponsor
- Project lead
- Project manager
- Data engineers
  - <i>preprocessing the necessary data into a form that data scientists can use</i>

- Data scientists
  - <i>Analyzing the data and developing a model for the project</i>

- DevOps
  - <i>Migrating the model and data preprocessing logics to the cloud </i>

  - <i>Supporting  software engineers with the deployment of the deliverable</i>

- Software engineers
  - <i>Developing the necessary tools for the project</i>
  - <i>Building the deliverable</i>
  - <i>Deploying the deliverable to the target s users</i>

  
**External stakeholders**
- Data collector
- Labeling company
- User
- C-suite executives
  
#### Task organization
實現目標的任務順序稱為關鍵路徑。

#### Resource allocation
對於深度學習專案來說，有兩種主要資源需要明確的資源分配：人力和運算資源。 人力資源是指積極致力於個人任務的員工。 一般來說，他們在data engineers、data scientists、DevOps 或software engineer領域擔任職位。

計算資源是指分配給專案的硬體和軟體資源。
