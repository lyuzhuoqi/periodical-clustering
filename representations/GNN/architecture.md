# Model Architecture

## WeightedGAT Architecture
```mermaid
graph TD
    subgraph Input
        I[Input Features, Edge Index, Edge Weights]
    end

    subgraph WeightEncoder
        WE[Weight Encoder]
        L1[Linear Layer<br/>1 → heads]
    end

    subgraph GATLayers
        GAT_IN[GAT Input Layer<br/>in_dim → hidden_dim/heads<br/>heads=4]
        GAT_OUT[GAT Output Layer<br/>in_dim → hidden_dim/heads<br/>heads=4]
        
        subgraph CombineLayer
            CAT[Concatenate]
            MLP1[Linear hidden_dim*2 → hidden_dim]
            RELU1[ReLU]
            MLP2[Linear hidden_dim → hidden_dim]
        end
        
        GAT2[Final GAT Layer<br/>hidden_dim → out_dim<br/>heads=1]
    end

    I --> WE
    WE --> |Edge Weights| GAT_IN
    WE --> |Edge Weights| GAT_OUT
    I --> |Node Features| GAT_IN
    I --> |Node Features| GAT_OUT
    I --> |Forward Edge Index| GAT_IN
    I --> |Flipped Edge Index| GAT_OUT
    
    GAT_IN --> |x1| CAT
    GAT_OUT --> |x2| CAT
    CAT --> MLP1
    MLP1 --> RELU1
    RELU1 --> MLP2
    MLP2 --> GAT2
    
    subgraph Output
        O[Node Embeddings]
    end
    
    GAT2 --> O
```

## WeightedBCELoss Computation Flow
```mermaid
graph TD
    subgraph Input
        P[Positive Scores]
        N[Negative Scores]
        EW[Edge Weights]
    end

    subgraph Processing
        C1[Concatenate Scores]
        C2[Create Labels<br/>pos=1, neg=0]
        W1[Calculate Mean Edge Weight]
        W2[Create Sample Weights]
    end

    subgraph Loss
        BCE[Binary Cross Entropy<br/>with Logits]
        NRM[Normalize by<br/>Weight Sum]
    end

    P --> C1
    N --> C1
    P --> C2
    N --> C2
    EW --> W1
    EW --> W2
    W1 --> W2
    
    C1 --> BCE
    C2 --> BCE
    W2 --> BCE
    BCE --> NRM
```

## Key Features

1. **WeightedGAT:**
   - Multi-head attention (4 heads by default)
   - Bidirectional information flow
   - Edge weight encoding
   - Dropout regularization
   - Two-stage GAT processing with MLP combination

2. **WeightedBCELoss:**
   - Weighted loss computation
   - Separate handling of positive and negative samples
   - Edge weight-based sample importance

3. **Training Features:**
   - Adam optimizer
   - ReduceLROnPlateau scheduler
   - Average Precision (AP) metric tracking
   - Negative edge sampling