

class Collider:
    def transform_by(self, transform: 'Transform3'):
        """Return a new Collider transformed by the given Transform3."""
        raise NotImplementedError("transform_by must be implemented by subclasses.")

    def closest_to_collider(self, other: "Collider"):
        """Return the closest points and distance between this collider and another collider."""
        raise NotImplementedError("closest_to_collider must be implemented by subclasses.")