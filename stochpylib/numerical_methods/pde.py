"""PDE tooling: meshes, finite-difference stencils and solvers (Poisson, heat,
Black-Scholes, advection), finite-element Galerkin (1-D and 2-D triangular P1/P2),
a FEniCS-flavoured adapter with a native fallback, boundary elements, and spectral
(Fourier/Chebyshev) methods.
"""

import xml.etree.ElementTree as ET

import numpy as np

from stochpylib.numerical_methods._common import _conjugate_gradient, _thomas
from stochpylib.numerical_methods.integration import GaussLegendre

__all__ = ["FiniteDifference", "FiniteElement", "FEniCS_Interface", "BoundaryElement",
          "SpectralMethod", "Mesh"]


class Mesh:
    """A minimal FEM mesh: node coordinates, element connectivity, boundary node indices."""

    def __init__(self, nodes, cells, boundary_nodes, dim):
        self.nodes = np.asarray(nodes, dtype=float)
        self.cells = np.asarray(cells, dtype=int)
        self.boundary_nodes = np.asarray(boundary_nodes, dtype=int)
        self.dim = int(dim)

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_cells(self):
        return self.cells.shape[0]

    @property
    def h(self):
        if self.dim == 1:
            return float(np.max(np.diff(self.nodes[:, 0] if self.nodes.ndim > 1 else self.nodes)))
        lengths = []
        for c in self.cells:
            pts = self.nodes[c]
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    lengths.append(np.linalg.norm(pts[i] - pts[j]))
        return float(np.max(lengths))

    @classmethod
    def interval(cls, a, b, n):
        nodes = np.linspace(a, b, n + 1)
        cells = np.column_stack([np.arange(n), np.arange(1, n + 1)])
        return cls(nodes, cells, [0, n], dim=1)

    @classmethod
    def rectangle(cls, x0, x1, y0, y1, nx, ny):
        xs = np.linspace(x0, x1, nx + 1)
        ys = np.linspace(y0, y1, ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        nodes = np.column_stack([X.ravel(), Y.ravel()])

        def idx(i, j):
            return i * (ny + 1) + j

        cells = []
        for i in range(nx):
            for j in range(ny):
                n00, n10 = idx(i, j), idx(i + 1, j)
                n01, n11 = idx(i, j + 1), idx(i + 1, j + 1)
                cells.append([n00, n10, n11])
                cells.append([n00, n11, n01])
        cells = np.array(cells)
        boundary = set()
        for i in range(nx + 1):
            boundary.add(idx(i, 0))
            boundary.add(idx(i, ny))
        for j in range(ny + 1):
            boundary.add(idx(0, j))
            boundary.add(idx(nx, j))
        return cls(nodes, cells, sorted(boundary), dim=2)

    def __repr__(self):
        return f"Mesh(dim={self.dim}, n_nodes={self.n_nodes}, n_cells={self.n_cells})"


class FiniteDifference:
    """Finite-difference stencils and classic PDE solvers on regular grids."""

    @staticmethod
    def stencil(order, offsets):
        """Fornberg finite-difference weights for derivative ``order`` at 0 on the given
        integer/float ``offsets`` (must include enough points for that order)."""
        offsets = np.asarray(offsets, dtype=float)
        n = len(offsets)
        if order >= n:
            raise ValueError("need more offsets than the derivative order")
        A = np.vander(offsets, n, increasing=True).T
        b = np.zeros(n)
        from math import factorial
        b[order] = factorial(order)
        return np.linalg.solve(A, b)

    def derivative(self, f, x, order=1, h=1e-4, accuracy=2, vectorized=False):
        from stochpylib.numerical_methods._common import _as_scalar_fn
        g = _as_scalar_fn(f, vectorized)
        n_pts = order + accuracy
        if n_pts % 2 == 0:
            n_pts += 1
        half = n_pts // 2
        offsets = np.arange(-half, half + 1)
        w = self.stencil(order, offsets)
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.zeros_like(x_arr)
        for wi, oi in zip(w, offsets):
            out += wi * g(x_arr + oi * h)
        out = out / h ** order
        return out[0] if np.ndim(x) == 0 else out

    def poisson_1d(self, f, a, b, n, bc=(0.0, 0.0)):
        """``-u'' = f`` on ``[a, b]`` with Dirichlet ``bc = (u(a), u(b))``."""
        x = np.linspace(a, b, n + 1)
        h = (b - a) / n
        from stochpylib.numerical_methods._common import _as_scalar_fn
        g = _as_scalar_fn(f, vectorized=False)
        interior = x[1:-1]
        rhs = g(interior) * h ** 2
        rhs[0] += bc[0]
        rhs[-1] += bc[1]
        lower = -np.ones(n - 2)
        diag = 2.0 * np.ones(n - 1)
        upper = -np.ones(n - 2)
        u_int = _thomas(lower, diag, upper, rhs)
        u = np.empty(n + 1)
        u[0], u[-1] = bc[0], bc[1]
        u[1:-1] = u_int
        return x, u

    def poisson_2d(self, f, bounds, nx, ny, bc=0.0, solver="cg", tol=1e-10):
        """``-Laplacian(u) = f`` on a rectangle, Dirichlet BC (constant or callable)."""
        (x0, x1), (y0, y1) = bounds
        hx = (x1 - x0) / nx
        hy = (y1 - y0) / ny
        xs = np.linspace(x0, x1, nx + 1)
        ys = np.linspace(y0, y1, ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")

        def bc_val(xv, yv):
            return bc(xv, yv) if callable(bc) else bc

        n_int_x, n_int_y = nx - 1, ny - 1
        N = n_int_x * n_int_y

        def idx(i, j):
            return (i - 1) * n_int_y + (j - 1)

        rhs = np.zeros(N)
        for i in range(1, nx):
            for j in range(1, ny):
                k = idx(i, j)
                rhs[k] = f(X[i, j], Y[i, j]) if callable(f) else f
                if i == 1:
                    rhs[k] += bc_val(xs[0], ys[j]) / hx ** 2
                if i == nx - 1:
                    rhs[k] += bc_val(xs[nx], ys[j]) / hx ** 2
                if j == 1:
                    rhs[k] += bc_val(xs[i], ys[0]) / hy ** 2
                if j == ny - 1:
                    rhs[k] += bc_val(xs[i], ys[ny]) / hy ** 2

        cx = 1.0 / hx ** 2
        cy = 1.0 / hy ** 2
        diag = 2 * cx + 2 * cy

        def matvec(v):
            V = v.reshape(n_int_x, n_int_y)
            out = diag * V
            out[1:, :] -= cx * V[:-1, :]
            out[:-1, :] -= cx * V[1:, :]
            out[:, 1:] -= cy * V[:, :-1]
            out[:, :-1] -= cy * V[:, 1:]
            return out.ravel()

        if solver == "dense":
            A = np.zeros((N, N))
            e = np.zeros(N)
            for k in range(N):
                e[:] = 0.0
                e[k] = 1.0
                A[:, k] = matvec(e)
            u_int = np.linalg.solve(A, rhs)
        else:
            u_int, _, _ = _conjugate_gradient(matvec, rhs, tol=tol)
        U = np.empty((nx + 1, ny + 1))
        for i in range(nx + 1):
            for j in range(ny + 1):
                if i == 0 or i == nx or j == 0 or j == ny:
                    U[i, j] = bc_val(xs[i], ys[j])
                else:
                    U[i, j] = u_int[idx(i, j)]
        return X, Y, U

    def heat_1d(self, u0, alpha, x_span, t_span, nx, nt, scheme="crank_nicolson", bc=(0.0, 0.0)):
        """``u_t = alpha * u_xx`` on ``x_span`` over ``t_span``, Dirichlet ``bc``."""
        a, b = x_span
        t0, tf = t_span
        x = np.linspace(a, b, nx + 1)
        dx = (b - a) / nx
        dt = (tf - t0) / nt
        r = alpha * dt / dx ** 2
        if scheme == "explicit" and r > 0.5:
            raise ValueError(f"explicit scheme requires r=alpha*dt/dx^2<=0.5, got {r:.3f}")
        U = np.empty((nt + 1, nx + 1))
        U[0] = u0(x) if callable(u0) else np.asarray(u0, dtype=float)
        for k in range(nt):
            u_old = U[k]
            u_new = np.empty(nx + 1)
            u_new[0], u_new[-1] = bc
            if scheme == "explicit":
                u_new[1:-1] = u_old[1:-1] + r * (u_old[2:] - 2 * u_old[1:-1] + u_old[:-2])
            else:
                theta = 1.0 if scheme == "implicit" else 0.5
                lower = -theta * r * np.ones(nx - 2)
                diag = (1 + 2 * theta * r) * np.ones(nx - 1)
                upper = -theta * r * np.ones(nx - 2)
                explicit_part = (1 - theta) * r * (u_old[2:] - 2 * u_old[1:-1] + u_old[:-2]) + u_old[1:-1]
                explicit_part[0] += theta * r * bc[0]
                explicit_part[-1] += theta * r * bc[1]
                u_new[1:-1] = _thomas(lower, diag, upper, explicit_part)
            U[k + 1] = u_new
        return x, np.linspace(t0, tf, nt + 1), U

    def black_scholes(self, K, T, r, sigma, S_max=None, n_S=200, n_t=200, kind="call",
                      scheme="crank_nicolson", american=False):
        """Backward finite-difference Black-Scholes PDE, on a uniform ``S`` grid."""
        if kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'")
        S_max = 4 * K if S_max is None else S_max
        dS = S_max / n_S
        dt = T / n_t
        S = np.linspace(0, S_max, n_S + 1)
        payoff = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
        V = payoff.copy()
        theta = 1.0 if scheme == "implicit" else 0.5 if scheme == "crank_nicolson" else 0.0
        j = np.arange(1, n_S)
        a = 0.5 * sigma ** 2 * j ** 2
        bcoef = 0.5 * r * j
        A_lower = -theta * dt * (a - bcoef)
        A_diag = 1 + theta * dt * (2 * a + r)
        A_upper = -theta * dt * (a + bcoef)
        B_lower = (1 - theta) * dt * (a - bcoef)
        B_diag = 1 - (1 - theta) * dt * (2 * a + r)
        B_upper = (1 - theta) * dt * (a + bcoef)
        for step in range(n_t):
            tau = (step + 1) * dt
            rhs = B_diag * V[1:-1]
            rhs[1:] += B_lower[1:] * V[1:-2]
            rhs[:-1] += B_upper[:-1] * V[2:-1]
            if kind == "call":
                v0, vN = 0.0, S_max - K * np.exp(-r * tau)
            else:
                v0, vN = K * np.exp(-r * tau), 0.0
            rhs[0] += theta * dt * (a[0] - bcoef[0]) * v0
            rhs[-1] += theta * dt * (a[-1] + bcoef[-1]) * vN
            V_new_int = _thomas(A_lower[1:], A_diag, A_upper[:-1], rhs)
            V = np.empty(n_S + 1)
            V[0], V[-1] = v0, vN
            V[1:-1] = V_new_int
            if american:
                V = np.maximum(V, payoff)
        self.S_ = S
        self.V_ = V
        return self

    def price(self, S0):
        return float(np.interp(S0, self.S_, self.V_))

    def advection_1d(self, u0, c, x_span, t_span, nx, nt, scheme="upwind", periodic=True):
        """``u_t + c u_x = 0`` on ``x_span``, periodic by default."""
        a, b = x_span
        t0, tf = t_span
        x = np.linspace(a, b, nx, endpoint=False) if periodic else np.linspace(a, b, nx)
        dx = (b - a) / nx if periodic else (b - a) / (nx - 1)
        dt = (tf - t0) / nt
        nu = c * dt / dx
        U = np.empty((nt + 1, len(x)))
        U[0] = u0(x) if callable(u0) else np.asarray(u0, dtype=float)
        for k in range(nt):
            u = U[k]
            if periodic:
                um1 = np.roll(u, 1)
                up1 = np.roll(u, -1)
            else:
                um1 = np.concatenate([[u[0]], u[:-1]])
                up1 = np.concatenate([u[1:], [u[-1]]])
            if scheme == "upwind":
                if c >= 0:
                    U[k + 1] = u - nu * (u - um1)
                else:
                    U[k + 1] = u - nu * (up1 - u)
            elif scheme == "lax_wendroff":
                U[k + 1] = u - 0.5 * nu * (up1 - um1) + 0.5 * nu ** 2 * (up1 - 2 * u + um1)
            else:
                raise ValueError("scheme must be 'upwind' or 'lax_wendroff'")
        return x, np.linspace(t0, tf, nt + 1), U

    def __repr__(self):
        return "FiniteDifference()"


class FiniteElement:
    """Galerkin finite elements: 1-D P1/P2 for ``-(p u')' + q u = f``, 2-D P1 for
    ``-Laplacian(u) = f`` on a triangular mesh.
    """

    def __init__(self, mesh, degree=1):
        self.mesh = mesh
        self.degree = int(degree)
        if mesh.dim == 2 and self.degree != 1:
            raise ValueError("2-D FiniteElement supports degree=1 only")

    def assemble(self, p=1.0, q=0.0, f=1.0, dirichlet=None, neumann=None):
        self.p, self.q, self.f = p, q, f
        # `or {}` would silently drop a genuine dirichlet=0.0 (falsy); check identity instead
        self.dirichlet = {} if dirichlet is None else dirichlet
        self.neumann = {} if neumann is None else neumann
        if self.mesh.dim == 1:
            self._assemble_1d()
        else:
            self._assemble_2d()
        return self

    def _pfun(self, x):
        return self.p(x) if callable(self.p) else self.p

    def _qfun(self, x):
        return self.q(x) if callable(self.q) else self.q

    def _ffun(self, x):
        return self.f(x) if callable(self.f) else self.f

    def _assemble_1d(self):
        mesh = self.mesh
        nodes_x = mesh.nodes if mesh.nodes.ndim == 1 else mesh.nodes[:, 0]
        n_nodes = mesh.n_nodes
        gl = GaussLegendre(4)
        if self.degree == 1:
            all_nodes = nodes_x
            n_dof = n_nodes
            elem_nodes_idx = mesh.cells
        else:
            edges = mesh.cells
            n_dof = n_nodes + len(edges)
            mid_x = 0.5 * (nodes_x[edges[:, 0]] + nodes_x[edges[:, 1]])
            all_nodes = np.concatenate([nodes_x, mid_x])
            elem_nodes_idx = np.column_stack([edges[:, 0], edges[:, 1], n_nodes + np.arange(len(edges))])
        K = np.zeros((n_dof, n_dof))
        F = np.zeros(n_dof)
        for e, idx in enumerate(elem_nodes_idx):
            xL, xR = nodes_x[mesh.cells[e, 0]], nodes_x[mesh.cells[e, 1]]
            h = xR - xL
            gp, gw = gl.nodes_weights(xL, xR)
            for xi, wi in zip(gp, gw):
                s = (xi - xL) / h
                if self.degree == 1:
                    N = np.array([1 - s, s])
                    dN = np.array([-1.0 / h, 1.0 / h])
                else:
                    N = np.array([2 * (s - 0.5) * (s - 1), 2 * s * (s - 0.5), -4 * s * (s - 1)])
                    dN = np.array([(4 * s - 3) / h, (4 * s - 1) / h, (-8 * s + 4) / h])
                pw = self._pfun(xi)
                qw = self._qfun(xi)
                fw = self._ffun(xi)
                for a in range(len(idx)):
                    F[idx[a]] += wi * fw * N[a]
                    for b in range(len(idx)):
                        K[idx[a], idx[b]] += wi * (pw * dN[a] * dN[b] + qw * N[a] * N[b])
        for side, val in self.neumann.items():
            node = 0 if side in (0, "left") else n_nodes - 1
            sign = -1.0 if node == 0 else 1.0
            F[node] += sign * val
        dirichlet_idx = {}
        for k, v in self.dirichlet.items():
            if k in ("left", 0):
                dirichlet_idx[0] = v
            elif k in ("right", -1, n_nodes - 1):
                dirichlet_idx[n_nodes - 1] = v
            else:
                dirichlet_idx[int(k)] = v
        # actual Dirichlet elimination happens in .solve() (needs the full unmodified
        # stiffness matrix to correctly subtract each column's contribution from F)
        self.stiffness_ = K
        self.load_ = F
        self.mass_ = None
        self.nodes_ = all_nodes
        self.dirichlet_idx_ = dirichlet_idx
        self._elem_nodes_idx = elem_nodes_idx
        self._nodes_x = nodes_x

    def _assemble_2d(self):
        mesh = self.mesh
        n = mesh.n_nodes
        K = np.zeros((n, n))
        M = np.zeros((n, n))
        F = np.zeros(n)
        for tri in mesh.cells:
            pts = mesh.nodes[tri]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            area = abs(area2) / 2.0
            b = np.array([y2 - y3, y3 - y1, y1 - y2])
            c = np.array([x3 - x2, x1 - x3, x2 - x1])
            Ke = (np.outer(b, b) + np.outer(c, c)) / (4 * area)
            centroid = pts.mean(axis=0)
            fval = self._ffun(centroid)
            Fe = np.full(3, fval * area / 3.0)
            Me = (area / 12.0) * (np.ones((3, 3)) + np.eye(3))
            for a in range(3):
                F[tri[a]] += Fe[a]
                for bb in range(3):
                    K[tri[a], tri[bb]] += Ke[a, bb]
                    M[tri[a], tri[bb]] += Me[a, bb]
        self.stiffness_ = K
        self.mass_ = M
        self.load_ = F
        self.nodes_ = mesh.nodes
        dirichlet_idx = {}
        for node in mesh.boundary_nodes:
            val = self.dirichlet(mesh.nodes[node]) if callable(self.dirichlet) else (
                self.dirichlet.get(int(node), self.dirichlet) if isinstance(self.dirichlet, dict)
                else self.dirichlet)
            dirichlet_idx[int(node)] = val
        self.dirichlet_idx_ = dirichlet_idx

    def solve(self):
        K = self.stiffness_.copy()
        F = self.load_.copy()
        n = K.shape[0]
        for node, val in self.dirichlet_idx_.items():
            F -= K[:, node] * val
            F[node] = val
            K[node, :] = 0.0
            K[:, node] = 0.0
            K[node, node] = 1.0
        self.u_ = np.linalg.solve(K, F)
        return self

    def evaluate(self, x):
        """Interpolate ``u_`` over the mesh using the element's own shape functions
        (linear for P1, quadratic using the midpoint DOF for P2) -- a P1-only linear
        lookup would silently discard the extra accuracy a P2 solve actually has.
        """
        mesh = self.mesh
        if mesh.dim != 1:
            raise NotImplementedError("2-D evaluate: use nearest-node lookup on nodes_ / u_")
        nodes_x = self._nodes_x
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        cell_idx = np.clip(np.searchsorted(nodes_x, x_arr, side="right") - 1, 0, len(nodes_x) - 2)
        out = np.empty_like(x_arr)
        for i, (xv, e) in enumerate(zip(x_arr, cell_idx)):
            xL, xR = nodes_x[mesh.cells[e, 0]], nodes_x[mesh.cells[e, 1]]
            h = xR - xL
            s = (xv - xL) / h
            idx = self._elem_nodes_idx[e]
            if self.degree == 1:
                N = np.array([1 - s, s])
            else:
                N = np.array([2 * (s - 0.5) * (s - 1), 2 * s * (s - 0.5), -4 * s * (s - 1)])
            out[i] = N @ self.u_[idx]
        return out[0] if np.ndim(x) == 0 else out

    def l2_error(self, exact):
        mesh = self.mesh
        if mesh.dim == 1:
            nodes_x = mesh.nodes if mesh.nodes.ndim == 1 else mesh.nodes[:, 0]
            xs = np.linspace(nodes_x[0], nodes_x[-1], 400)
            u_h = self.evaluate(xs)
            u_ex = np.array([exact(xi) for xi in xs])
            return float(np.sqrt(np.trapezoid((u_h - u_ex) ** 2, xs)))
        total = 0.0
        for tri in mesh.cells:
            pts = mesh.nodes[tri]
            centroid = pts.mean(axis=0)
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0
            u_h = self.u_[tri].mean()
            u_ex = exact(centroid)
            total += (u_h - u_ex) ** 2 * area
        return float(np.sqrt(total))

    @property
    def h(self):
        return self.mesh.h

    def __repr__(self):
        return f"FiniteElement(dim={self.mesh.dim}, degree={self.degree})"


class FEniCS_Interface:
    """FEniCS-flavoured problem builder: solves natively via :class:`FiniteElement` by
    default, and can lazily hand off mesh construction to a real dolfinx/dolfin install
    if present (``.to_fenics()``), or export the mesh in DOLFIN-XML/XDMF for one.
    """

    def __init__(self, mesh, family="Lagrange", degree=1):
        self.mesh = mesh
        self.family = family
        self.degree = int(degree)
        self._dirichlet = {}
        self._diffusion = 1.0
        self._reaction = 0.0
        self._source = 0.0

    def dirichlet_bc(self, value, where=None):
        if self.mesh.dim == 1:
            if where is None:
                self._dirichlet = {"left": value, "right": value}
            else:
                self._dirichlet = where if isinstance(where, dict) else {"left": value}
        else:
            if where is None:
                self._dirichlet = value
            else:
                self._dirichlet = {int(i): value(self.mesh.nodes[i]) if callable(value) else value
                                   for i in self.mesh.boundary_nodes if where(self.mesh.nodes[i])}
        return self

    def set_equation(self, diffusion=1.0, reaction=0.0, source=0.0):
        self._diffusion = diffusion
        self._reaction = reaction
        self._source = source
        return self

    def solve(self):
        fe = FiniteElement(self.mesh, degree=self.degree)
        if self.mesh.dim == 1:
            fe.assemble(p=self._diffusion, q=self._reaction, f=self._source,
                       dirichlet=self._dirichlet)
        else:
            fe.assemble(f=self._source, dirichlet=self._dirichlet)
        fe.solve()
        self.u_ = fe.u_
        self.nodes_ = fe.nodes_
        self._fe = fe
        return self

    def export_mesh(self, path, format="dolfin_xml"):
        mesh = self.mesh
        if format == "dolfin_xml":
            _write_dolfin_xml(mesh, path)
        elif format == "xdmf":
            _write_xdmf(mesh, path)
        else:
            raise ValueError("format must be 'dolfin_xml' or 'xdmf'")
        return self

    @staticmethod
    def read_mesh(path):
        """Round-trip reader for the DOLFIN-XML export (pure ``xml.etree``, no FEniCS)."""
        tree = ET.parse(path)
        root = tree.getroot()
        mesh_el = root.find("mesh")
        celltype = mesh_el.get("celltype")
        dim = int(mesh_el.get("dim"))
        verts_el = mesh_el.find("vertices")
        n_v = int(verts_el.get("size"))
        nodes = np.zeros((n_v, dim)) if dim > 1 else np.zeros(n_v)
        for v in verts_el.findall("vertex"):
            i = int(v.get("index"))
            if dim == 1:
                nodes[i] = float(v.get("x"))
            else:
                nodes[i] = [float(v.get("x")), float(v.get("y"))]
        cells_el = mesh_el.find("cells")
        n_c = int(cells_el.get("size"))
        tag = "interval" if celltype == "interval" else "triangle"
        n_local = 2 if celltype == "interval" else 3
        cells = np.zeros((n_c, n_local), dtype=int)
        for c in cells_el.findall(tag):
            i = int(c.get("index"))
            cells[i] = [int(c.get(f"v{k}")) for k in range(n_local)]
        if celltype == "interval":
            boundary = [0, n_v - 1]
        else:
            from collections import Counter
            edge_count = Counter()
            for tri in cells:
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    edge_count[frozenset((a, b))] += 1
            boundary = sorted({n for e, cnt in edge_count.items() if cnt == 1 for n in e})
        return Mesh(nodes, cells, boundary, dim=dim)

    def to_fenics(self):
        """Build the mesh through a real FEniCS install (dolfinx preferred, then
        legacy dolfin). Raises ``ImportError`` with a clear message if neither is
        installed -- stochpylib never depends on FEniCS; use :meth:`solve` for the
        native solver instead.
        """
        import importlib

        mesh = self.mesh
        try:
            dolfinx = importlib.import_module("dolfinx")
        except ImportError:
            dolfinx = None
        if dolfinx is not None:
            from mpi4py import MPI
            if mesh.dim == 1:
                a, b = float(mesh.nodes[0]), float(mesh.nodes[-1])
                n = mesh.n_cells
                dmesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, n, [a, b])
            else:
                x0, y0 = mesh.nodes.min(axis=0)
                x1, y1 = mesh.nodes.max(axis=0)
                nx = ny = int(np.sqrt(mesh.n_cells / 2))
                dmesh = dolfinx.mesh.create_rectangle(
                    MPI.COMM_WORLD, [[x0, y0], [x1, y1]], [max(nx, 1), max(ny, 1)])
            return {"mesh": dmesh, "backend": "dolfinx"}
        try:
            dolfin = importlib.import_module("dolfin")
        except ImportError:
            raise ImportError(
                "FEniCS (dolfinx/dolfin) is not installed. stochpylib never depends on it; "
                "use .solve() for the native solver or install FEniCS separately."
            )
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mesh.xml")
            self.export_mesh(path, format="dolfin_xml")
            dmesh = dolfin.Mesh(path)
        return {"mesh": dmesh, "backend": "dolfin"}

    def __repr__(self):
        return f"FEniCS_Interface(dim={self.mesh.dim}, family={self.family!r}, degree={self.degree})"


def _write_dolfin_xml(mesh, path):
    dim = mesh.dim
    celltype = "interval" if dim == 1 else "triangle"
    nodes = mesh.nodes if mesh.nodes.ndim > 1 else mesh.nodes[:, None]
    lines = ['<?xml version="1.0"?>',
            '<dolfin xmlns:dolfin="http://fenicsproject.org">',
            f'  <mesh celltype="{celltype}" dim="{dim}">',
            f'    <vertices size="{mesh.n_nodes}">']
    for i, row in enumerate(nodes):
        if dim == 1:
            lines.append(f'      <vertex index="{i}" x="{float(row[0]):.17g}"/>')
        else:
            lines.append(f'      <vertex index="{i}" x="{float(row[0]):.17g}" y="{float(row[1]):.17g}"/>')
    lines.append('    </vertices>')
    lines.append(f'    <cells size="{mesh.n_cells}">')
    for i, c in enumerate(mesh.cells):
        if dim == 1:
            lines.append(f'      <interval index="{i}" v0="{c[0]}" v1="{c[1]}"/>')
        else:
            lines.append(f'      <triangle index="{i}" v0="{c[0]}" v1="{c[1]}" v2="{c[2]}"/>')
    lines.append('    </cells>')
    lines.append('  </mesh>')
    lines.append('</dolfin>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _write_xdmf(mesh, path):
    dim = mesh.dim
    nodes = mesh.nodes if mesh.nodes.ndim > 1 else mesh.nodes[:, None]
    topo_type = "polyline" if dim == 1 else "triangle"
    node_str = " ".join(" ".join(f"{v:.16g}" for v in row) for row in nodes)
    cell_str = " ".join(" ".join(str(v) for v in row) for row in mesh.cells)
    geom_dim = nodes.shape[1]
    xml = f'''<?xml version="1.0"?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="mesh" GridType="Uniform">
      <Topology TopologyType="{topo_type}" NumberOfElements="{mesh.n_cells}">
        <DataItem Format="XML" Dimensions="{mesh.n_cells} {mesh.cells.shape[1]}">{cell_str}</DataItem>
      </Topology>
      <Geometry GeometryType="{"XY" if geom_dim == 2 else "X"}">
        <DataItem Format="XML" Dimensions="{mesh.n_nodes} {geom_dim}">{node_str}</DataItem>
      </Geometry>
    </Grid>
  </Domain>
</Xdmf>
'''
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)


class BoundaryElement:
    """2-D interior Laplace problem via the direct boundary-element method (constant
    elements, fundamental solution ``u* = -(1/2*pi) * ln r``).
    """

    def __init__(self, boundary_points):
        pts = np.asarray(boundary_points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("boundary_points must be (n, 2)")
        self.points = pts
        n = len(pts)
        self.midpoints_ = 0.5 * (pts + np.roll(pts, -1, axis=0))
        edge = np.roll(pts, -1, axis=0) - pts
        self.lengths_ = np.linalg.norm(edge, axis=1)
        normal = np.column_stack([edge[:, 1], -edge[:, 0]])
        self.normals_ = normal / self.lengths_[:, None]

    def _assemble(self):
        n = len(self.midpoints_)
        H = np.zeros((n, n))
        G = np.zeros((n, n))
        gl = GaussLegendre(8)
        for i in range(n):
            xi = self.midpoints_[i]
            for j in range(n):
                if i == j:
                    H[i, j] = 0.5
                    L = self.lengths_[j]
                    G[i, j] = (L / (2 * np.pi)) * (1.0 - np.log(L / 2.0))
                else:
                    p0 = self.points[j]
                    p1 = self.points[(j + 1) % n]
                    L = self.lengths_[j]
                    nvec = self.normals_[j]
                    t, w = gl.nodes_weights(0.0, 1.0)
                    pts_on = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
                    r_vec = pts_on - xi[None, :]
                    r = np.linalg.norm(r_vec, axis=1)
                    r = np.maximum(r, 1e-14)
                    dudn = -(r_vec @ nvec) / (2 * np.pi * r ** 2)
                    H[i, j] = float(np.sum(w * dudn)) * L
                    G[i, j] = -float(np.sum(w * np.log(r) / (2 * np.pi))) * L
        self.H_, self.G_ = H, G
        return H, G

    def solve(self, dirichlet):
        H, G = self._assemble()
        n = len(self.midpoints_)
        u = dirichlet(self.midpoints_) if callable(dirichlet) else np.asarray(dirichlet, dtype=float)
        u = np.broadcast_to(u, (n,)).astype(float)
        rhs = H @ u
        q = np.linalg.solve(G, rhs)
        self.flux_ = q
        self.u_boundary_ = u
        return self

    def evaluate(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        n = len(self.midpoints_)
        gl = GaussLegendre(8)
        out = np.zeros(len(points))
        for k, x in enumerate(points):
            total = 0.0
            for j in range(n):
                p0 = self.points[j]
                p1 = self.points[(j + 1) % n]
                nvec = self.normals_[j]
                t, w = gl.nodes_weights(0.0, 1.0)
                pts_on = p0[None, :] + t[:, None] * (p1 - p0)[None, :]
                r_vec = pts_on - x[None, :]
                r = np.maximum(np.linalg.norm(r_vec, axis=1), 1e-14)
                L = self.lengths_[j]
                Gj = -float(np.sum(w * np.log(r) / (2 * np.pi))) * L
                dudn = -(r_vec @ nvec) / (2 * np.pi * r ** 2)
                Hj = float(np.sum(w * dudn)) * L
                total += Gj * self.flux_[j] - Hj * self.u_boundary_[j]
            out[k] = total
        return out

    def __repr__(self):
        return f"BoundaryElement(n_elements={len(self.midpoints_)})"


class SpectralMethod:
    """Fourier (periodic) and Chebyshev-collocation spectral tools."""

    @staticmethod
    def fourier_derivative(u, L, order=1):
        n = len(u)
        k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
        u_hat = np.fft.fft(u)
        deriv_hat = (1j * k) ** order * u_hat
        if order % 2 == 1 and n % 2 == 0:
            deriv_hat[n // 2] = 0.0  # zero the Nyquist mode for odd-order derivatives
        return np.real(np.fft.ifft(deriv_hat))

    @staticmethod
    def poisson_periodic(f, L, dim=1):
        if dim == 1:
            n = len(f)
            k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
            f_hat = np.fft.fft(f)
            u_hat = np.zeros_like(f_hat)
            nz = k != 0
            u_hat[nz] = f_hat[nz] / (k[nz] ** 2)
            return np.real(np.fft.ifft(u_hat))
        f = np.asarray(f)
        nx, ny = f.shape
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=L / nx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=L / ny)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K2 = KX ** 2 + KY ** 2
        f_hat = np.fft.fft2(f)
        u_hat = np.zeros_like(f_hat)
        nz = K2 != 0
        u_hat[nz] = f_hat[nz] / K2[nz]
        return np.real(np.fft.ifft2(u_hat))

    @staticmethod
    def heat_periodic(u0, alpha, L, T):
        n = len(u0)
        k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
        u0_hat = np.fft.fft(u0)
        u_hat = u0_hat * np.exp(-alpha * k ** 2 * T)
        return np.real(np.fft.ifft(u_hat))

    @staticmethod
    def chebyshev_diff_matrix(n):
        """Trefethen's ``cheb(N)``: returns ``(D, x)`` on ``[-1, 1]``, ``N = n``."""
        if n == 0:
            return np.array([[0.0]]), np.array([1.0])
        x = np.cos(np.pi * np.arange(n + 1) / n)
        c = np.ones(n + 1)
        c[0] = 2.0
        c[-1] = 2.0
        c *= (-1.0) ** np.arange(n + 1)
        X = np.tile(x, (n + 1, 1)).T
        dX = X - X.T
        D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
        D -= np.diag(D.sum(axis=1))
        return D, x

    def solve_bvp(self, f, a, b, n, bc=(0.0, 0.0), p=0.0, q=0.0):
        """``u'' + p*u' + q*u = f`` on ``[a, b]`` with Dirichlet ``bc``, Chebyshev collocation."""
        D, x_std = self.chebyshev_diff_matrix(n)
        x = 0.5 * (a + b) + 0.5 * (b - a) * x_std
        scale = 2.0 / (b - a)
        D1 = D * scale
        D2 = D1 @ D1
        L = D2 + p * D1 + q * np.eye(n + 1)
        rhs = f(x) if callable(f) else np.asarray(f, dtype=float)
        rhs = np.array(rhs, dtype=float, copy=True)
        L = L.copy()
        # boundary rows: x[0] is +1 end (b), x[n] is -1 end (a) per Trefethen's ordering
        L[0, :] = 0.0
        L[0, 0] = 1.0
        rhs[0] = bc[1]
        L[-1, :] = 0.0
        L[-1, -1] = 1.0
        rhs[-1] = bc[0]
        u = np.linalg.solve(L, rhs)
        return x, u

    def __repr__(self):
        return "SpectralMethod()"
