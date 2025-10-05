class ART1:
    """
    Adaptive Resonance Theory (ART1) for binary data clustering

    Attributes:
        num_features (int): Dimensionality of input vectors
        vigilance (float): Vigilance parameter (0 < rho <= 1.0)
        L (float): Learning parameter (L > 1.0)
        max_clusters (int): Maximum clusters allowed (None for unlimited)
        templates (list): Cluster prototype vectors (top-down weights)
        cluster_history (list): Creation order of clusters
    """

    def __init__(self, num_features, vigilance=0.7, L=1.1, max_clusters=None):
        """
        Initialize ART1 clustering model

        Args:
            num_features: Dimension of binary input vectors
            vigilance: Vigilance parameter (0.7 typical)
            L: Learning parameter >1.0 (1.1 typical)
            max_clusters: Maximum clusters to create (None = unlimited)
        """
        if not 0 < vigilance <= 1.0:
            raise ValueError("Vigilance must be in (0, 1]")
        if L <= 1.0:
            raise ValueError("L must be > 1.0")

        self.num_features = num_features
        self.vigilance = vigilance
        self.L = L
        self.max_clusters = max_clusters
        self.templates = []
        self.cluster_history = []

    def _calculate_choice_function(self, x, template):
        """
        Calculate choice function (T_j) for input vector x and cluster template

        T_j = |x ∧ w_j| * L / (L - 1 + |w_j|)

        Args:
            x: Input vector (binary list)
            template: Cluster template vector

        Returns:
            T_j: Choice function value
            intersection: |x ∧ w_j|
        """
        intersection = sum(x_i & w_i for x_i, w_i in zip(x, template))
        template_norm = sum(template)
        denominator = self.L - 1 + template_norm

        if template_norm == 0:  # Avoid division by zero
            return 0.0, intersection
        return (self.L * intersection) / denominator, intersection

    def train(self, data):
        """
        Train ART1 model on binary data

        Args:
            data: Iterable of binary vectors (each vector length = num_features)
        """
        for x in data:
            self._train_pattern(x)

    def _train_pattern(self, x):
        """Process a single input pattern during training"""
        # Skip zero vectors
        if sum(x) == 0:
            return

        # Initialize first cluster if none exist
        if not self.templates:
            self._create_new_cluster(x)
            return

        # Calculate choice function for all clusters
        candidates = []
        for idx, template in enumerate(self.templates):
            T_j, intersection = self._calculate_choice_function(x, template)
            candidates.append((idx, T_j, intersection))

        # Sort by choice function descending
        candidates.sort(key=lambda c: c[1], reverse=True)

        # Find best matching cluster meeting vigilance
        matched = False
        x_norm = sum(x)
        for idx, T_j, intersection in candidates:
            similarity = intersection / x_norm
            if similarity >= self.vigilance:
                self._update_cluster(idx, x)
                matched = True
                break

        # Create new cluster if no match found
        if not matched:
            self._create_new_cluster(x)

    def _update_cluster(self, cluster_idx, x):
        """Update cluster template with new input vector"""
        # Update template: w_j ∧ x
        self.templates[cluster_idx] = [
            x_i & w_i for x_i, w_i in zip(x, self.templates[cluster_idx])
        ]

    def _create_new_cluster(self, x):
        """Create new cluster from input vector"""
        if self.max_clusters and len(self.templates) >= self.max_clusters:
            return

        self.templates.append(x[:])  # Store copy of vector
        self.cluster_history.append(len(self.templates) - 1)

    def predict(self, x):
        """
        Predict cluster assignment for input vector

        Args:
            x: Binary input vector

        Returns:
            Cluster index or -1 if no matching cluster
        """
        if sum(x) == 0:  # Zero vector never matches
            return -1

        if not self.templates:
            return -1

        # Calculate choice function for all clusters
        candidates = []
        for idx, template in enumerate(self.templates):
            T_j, intersection = self._calculate_choice_function(x, template)
            candidates.append((idx, T_j, intersection))

        # Sort by choice function descending
        candidates.sort(key=lambda c: c[1], reverse=True)

        # Return first cluster meeting vigilance
        x_norm = sum(x)
        for idx, T_j, intersection in candidates:
            similarity = intersection / x_norm
            if similarity >= self.vigilance:
                return idx

        return -1  # No cluster found

    @property
    def n_clusters(self):
        """Number of clusters currently in model"""
        return len(self.templates)
