import { NavLink } from "react-router-dom";

function Layout({ children }) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">P</div>

          <div>
            <h1>PII Privacy</h1>
            <span>Intelligence Platform</span>
          </div>
        </div>

        <nav className="navigation">
          <NavLink
            to="/upload"
            className={({ isActive }) =>
              `nav-item ${isActive ? "active" : ""}`
            }
          >
            <span>↑</span>
            Upload
          </NavLink>

          <NavLink
            to="/results"
            className={({ isActive }) =>
              `nav-item ${isActive ? "active" : ""}`
            }
          >
            <span>▣</span>
            Results
          </NavLink>

          <NavLink
            to="/analytics"
            className={({ isActive }) =>
              `nav-item ${isActive ? "active" : ""}`
            }
          >
            <span>◒</span>
            Analytics
          </NavLink>
        </nav>

        <div className="sidebar-footer">
          <div className="status-dot"></div>

          <div>
            <strong>System Online</strong>
            <span>PII engine ready</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">PRIVACY INTELLIGENCE</p>
            <h2>PII Detection & Analysis</h2>
          </div>

          <div className="topbar-status">
            <span className="status-dot"></span>
            API Connected
          </div>
        </header>

        <section className="page-content">
          {children}
        </section>
      </main>
    </div>
  );
}

export default Layout;