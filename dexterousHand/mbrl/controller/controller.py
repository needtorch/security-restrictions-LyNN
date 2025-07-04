import torch

class Controller:
    """Minimal placeholder Controller class containing the MPC cost function.

    NOTE: This is a stub implementation created to ensure that the codebase
    has a `mpc_cost_fun` method with the correct device-handling logic. It
    should be replaced or extended with the full controller implementation as
    required by the project.
    """

    @staticmethod
    def mpc_cost_fun(value: torch.Tensor, c, *args, **kwargs):
        """Compute MPC cost and return whether the value exceeds a threshold.

        Parameters
        ----------
        value : torch.Tensor
            Tensor holding the evaluated value(s).
        c : float or torch.Tensor
            Threshold against which `value` is compared. Will be moved to the
            same device as `value` before the comparison.
        *args, **kwargs
            Additional parameters (ignored in this stub).

        Returns
        -------
        torch.Tensor
            Boolean tensor (`torch.bool`) indicating which elements are above
            the threshold.
        """
        # --- Device handling fix ------------------------------------------------
        # Ensure `c` lives on the same device as `value` before the comparison
        if torch.is_tensor(c):
            c = c.to(value.device, dtype=value.dtype)
        else:
            c = torch.tensor(c, device=value.device, dtype=value.dtype)

        over_threshold = value > c  # shape broadcast as usual
        return over_threshold