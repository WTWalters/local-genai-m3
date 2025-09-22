// Orthopedics EMR Frontend Application
class MedicalEMR {
    constructor() {
        this.currentUser = null;
        this.currentRole = null;
        this.apiBase = '/api/v1';
        this.init();
    }

    init() {
        this.bindEvents();
        this.showScreen('loginScreen');
    }

    bindEvents() {
        // Login form
        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        }

        // Demo role buttons
        document.querySelectorAll('.btn-demo').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleDemoLogin(e));
        });

        // Logout button
        const logoutBtn = document.getElementById('logoutBtn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => this.handleLogout());
        }

        // Toast close buttons
        document.querySelectorAll('.toast-close').forEach(btn => {
            btn.addEventListener('click', (e) => this.hideToast(e.target.closest('.toast')));
        });
    }

    // Screen Management
    showScreen(screenId) {
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        document.getElementById(screenId).classList.add('active');
    }

    // Authentication
    async handleLogin(e) {
        e.preventDefault();
        const formData = new FormData(e.target);
        const username = formData.get('username');
        const password = formData.get('password');

        try {
            this.showScreen('loadingScreen');
            const response = await fetch(`${this.apiBase}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            if (response.ok) {
                const data = await response.json();
                this.setCurrentUser(data.user, data.access_token);
                this.showDashboard();
            } else {
                const error = await response.json();
                this.showError(error.detail?.message || 'Login failed');
                this.showScreen('loginScreen');
            }
        } catch (error) {
            this.showError('Connection error. Please try again.');
            this.showScreen('loginScreen');
        }
    }

    async handleDemoLogin(e) {
        const role = e.target.dataset.role;
        
        try {
            this.showScreen('loadingScreen');
            
            // Simulate API call to get role permissions
            const response = await fetch(`${this.apiBase}/test-rbac/test-permissions/${role}`);
            
            if (response.ok) {
                const data = await response.json();
                
                // Create mock user for demo
                const mockUser = {
                    username: `demo_${role}`,
                    role: role,
                    permissions: data.permissions,
                    can_access_phi: data.can_access_phi
                };
                
                this.setCurrentUser(mockUser, 'demo-token');
                this.showDashboard();
                this.showSuccess(`Logged in as ${role.replace('_', ' ').toUpperCase()}`);
            } else {
                this.showError('Failed to load demo role');
                this.showScreen('loginScreen');
            }
        } catch (error) {
            this.showError('Connection error. Please try again.');
            this.showScreen('loginScreen');
        }
    }

    handleLogout() {
        this.currentUser = null;
        this.currentRole = null;
        localStorage.removeItem('emr_user');
        localStorage.removeItem('emr_token');
        this.showScreen('loginScreen');
        this.showSuccess('Logged out successfully');
    }

    setCurrentUser(user, token) {
        this.currentUser = user;
        this.currentRole = user.role;
        
        localStorage.setItem('emr_user', JSON.stringify(user));
        localStorage.setItem('emr_token', token);
        
        this.updateUserDisplay();
    }

    updateUserDisplay() {
        const userElement = document.getElementById('currentUser');
        const roleElement = document.getElementById('currentRole');
        
        if (userElement && this.currentUser) {
            userElement.textContent = this.currentUser.username;
        }
        
        if (roleElement && this.currentRole) {
            roleElement.textContent = this.currentRole.replace('_', ' ').toUpperCase();
            roleElement.className = `role-badge ${this.currentRole}`;
        }
    }

    // Dashboard
    showDashboard() {
        this.showScreen('dashboardScreen');
        this.buildSidebar();
        this.showWelcome();
    }

    buildSidebar() {
        const sidebarMenu = document.getElementById('sidebarMenu');
        if (!sidebarMenu || !this.currentUser) return;

        const menuStructure = this.getMenuForRole(this.currentRole);
        sidebarMenu.innerHTML = this.buildMenuHTML(menuStructure);
        
        // Bind menu clicks
        sidebarMenu.addEventListener('click', (e) => {
            const menuItem = e.target.closest('.menu-item');
            if (menuItem) {
                e.preventDefault();
                this.handleMenuClick(menuItem);
            }
        });
    }

    getMenuForRole(role) {
        const baseMenu = [
            {
                title: 'Dashboard',
                items: [
                    { id: 'overview', icon: 'fas fa-tachometer-alt', label: 'Overview', action: 'showOverview' }
                ]
            }
        ];

        const roleMenus = {
            admin: [
                ...baseMenu,
                {
                    title: 'Administration',
                    items: [
                        { id: 'users', icon: 'fas fa-users', label: 'User Management', action: 'showUsers' },
                        { id: 'system', icon: 'fas fa-cogs', label: 'System Settings', action: 'showSystem' },
                        { id: 'audit', icon: 'fas fa-clipboard-list', label: 'Audit Logs', action: 'showAudit' }
                    ]
                },
                {
                    title: 'Data',
                    items: [
                        { id: 'phi-data', icon: 'fas fa-database', label: 'PHI Data Access', action: 'showPHI' },
                        { id: 'analytics', icon: 'fas fa-chart-bar', label: 'Analytics', action: 'showAnalytics' }
                    ]
                }
            ],
            attending_physician: [
                ...baseMenu,
                {
                    title: 'Patient Care',
                    items: [
                        { id: 'patients', icon: 'fas fa-user-injured', label: 'My Patients', action: 'showPatients' },
                        { id: 'phi-data', icon: 'fas fa-notes-medical', label: 'PHI Access', action: 'showPHI' },
                        { id: 'documents', icon: 'fas fa-file-medical', label: 'Medical Documents', action: 'showDocuments' }
                    ]
                },
                {
                    title: 'Tools',
                    items: [
                        { id: 'query', icon: 'fas fa-search', label: 'Advanced Queries', action: 'showQuery' },
                        { id: 'analytics', icon: 'fas fa-chart-line', label: 'Analytics', action: 'showAnalytics' }
                    ]
                }
            ],
            resident: [
                ...baseMenu,
                {
                    title: 'Patient Care',
                    items: [
                        { id: 'patients', icon: 'fas fa-user-injured', label: 'Assigned Patients', action: 'showPatients' },
                        { id: 'phi-data', icon: 'fas fa-notes-medical', label: 'PHI Access', action: 'showPHI' },
                        { id: 'documents', icon: 'fas fa-file-medical', label: 'Documentation', action: 'showDocuments' }
                    ]
                },
                {
                    title: 'Learning',
                    items: [
                        { id: 'query', icon: 'fas fa-search', label: 'Medical Queries', action: 'showQuery' }
                    ]
                }
            ],
            nurse: [
                ...baseMenu,
                {
                    title: 'Patient Care',
                    items: [
                        { id: 'patients', icon: 'fas fa-user-injured', label: 'Patient Care', action: 'showPatients' },
                        { id: 'phi-data', icon: 'fas fa-notes-medical', label: 'PHI Access', action: 'showPHI' },
                        { id: 'documents', icon: 'fas fa-file-medical', label: 'Nursing Notes', action: 'showDocuments' }
                    ]
                }
            ],
            read_only: [
                ...baseMenu,
                {
                    title: 'Information',
                    items: [
                        { id: 'phi-data', icon: 'fas fa-eye', label: 'View PHI Data', action: 'showPHI' },
                        { id: 'documents', icon: 'fas fa-file-alt', label: 'View Documents', action: 'showDocuments' }
                    ]
                }
            ]
        };

        return roleMenus[role] || baseMenu;
    }

    buildMenuHTML(menuStructure) {
        return menuStructure.map(section => `
            <div class="menu-section">
                <div class="menu-section-title">${section.title}</div>
                ${section.items.map(item => `
                    <a href="#" class="menu-item" data-action="${item.action}" data-id="${item.id}">
                        <i class="${item.icon}"></i>
                        ${item.label}
                    </a>
                `).join('')}
            </div>
        `).join('');
    }

    handleMenuClick(menuItem) {
        // Remove active class from all items
        document.querySelectorAll('.menu-item').forEach(item => {
            item.classList.remove('active');
        });
        
        // Add active class to clicked item
        menuItem.classList.add('active');
        
        const action = menuItem.dataset.action;
        const id = menuItem.dataset.id;
        
        if (this[action]) {
            this[action](id);
        }
    }

    // Content Panels
    showOverview() {
        this.updateContent('Dashboard Overview', 'Home > Dashboard', `
            <div class="welcome-panel">
                <h2>Welcome, ${this.currentUser.username}!</h2>
                <p>Role: <strong>${this.currentRole.replace('_', ' ').toUpperCase()}</strong></p>
                <p>You have access to ${this.currentUser.permissions.length} permissions.</p>
            </div>
            
            <div class="info-panel">
                <h3><i class="fas fa-shield-alt"></i> Your Permissions</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 10px;">
                    ${this.currentUser.permissions.slice(0, 12).map(perm => `
                        <span style="background: var(--light-gray); padding: 5px 10px; border-radius: 4px; font-size: 0.9rem;">
                            ${perm}
                        </span>
                    `).join('')}
                    ${this.currentUser.permissions.length > 12 ? `<span style="color: var(--medium-gray);">... and ${this.currentUser.permissions.length - 12} more</span>` : ''}
                </div>
            </div>

            ${this.currentUser.can_access_phi ? `
                <div class="warning-panel">
                    <h3><i class="fas fa-exclamation-triangle"></i> PHI Access</h3>
                    <p>You have access to Protected Health Information. Please ensure compliance with HIPAA regulations.</p>
                </div>
            ` : ''}
        `);
    }

    async showPHI() {
        this.updateContent('PHI Data Access', 'Home > PHI Access', '<div class="loading-spinner"><i class="fas fa-stethoscope fa-spin"></i></div>');
        
        try {
            const response = await fetch(`${this.apiBase}/test-rbac/phi-read`);
            
            if (response.ok) {
                const data = await response.json();
                this.updateContent('PHI Data Access', 'Home > PHI Access', `
                    <div class="success-panel">
                        <h3><i class="fas fa-check-circle"></i> PHI Access Granted</h3>
                        <p>${data.message}</p>
                    </div>
                    
                    <div class="info-panel">
                        <h3>Sample PHI Data</h3>
                        <pre style="background: var(--light-gray); padding: 15px; border-radius: 4px; margin-top: 10px;">
${JSON.stringify(data.sample_phi_data, null, 2)}
                        </pre>
                    </div>

                    <div class="warning-panel">
                        <h3><i class="fas fa-shield-alt"></i> HIPAA Compliance Notice</h3>
                        <p>This data contains Protected Health Information (PHI). Access is logged and monitored for compliance purposes.</p>
                    </div>
                `);
            } else {
                const error = await response.json();
                this.updateContent('PHI Data Access', 'Home > PHI Access', `
                    <div class="warning-panel">
                        <h3><i class="fas fa-times-circle"></i> Access Denied</h3>
                        <p>${error.error || 'You do not have permission to access PHI data.'}</p>
                    </div>
                `);
            }
        } catch (error) {
            this.showError('Failed to load PHI data');
        }
    }

    showUsers() {
        this.updateContent('User Management', 'Home > Administration > Users', `
            <div class="info-panel">
                <h3><i class="fas fa-users"></i> User Management</h3>
                <p>Manage medical staff accounts and permissions.</p>
            </div>
            
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Username</th>
                        <th>Role</th>
                        <th>Department</th>
                        <th>Status</th>
                        <th>Last Login</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>dr_smith</td>
                        <td><span class="role-badge attending_physician">Attending Physician</span></td>
                        <td>Orthopedics</td>
                        <td><span style="color: var(--medical-green);">Active</span></td>
                        <td>2 hours ago</td>
                        <td>
                            <button style="padding: 4px 8px; border: 1px solid var(--border-color); background: var(--white); border-radius: 4px; margin-right: 5px;">Edit</button>
                            <button style="padding: 4px 8px; border: 1px solid var(--danger-color); background: var(--white); color: var(--danger-color); border-radius: 4px;">Disable</button>
                        </td>
                    </tr>
                    <tr>
                        <td>nurse_jones</td>
                        <td><span class="role-badge nurse">Nurse</span></td>
                        <td>Orthopedics</td>
                        <td><span style="color: var(--medical-green);">Active</span></td>
                        <td>1 day ago</td>
                        <td>
                            <button style="padding: 4px 8px; border: 1px solid var(--border-color); background: var(--white); border-radius: 4px; margin-right: 5px;">Edit</button>
                            <button style="padding: 4px 8px; border: 1px solid var(--danger-color); background: var(--white); color: var(--danger-color); border-radius: 4px;">Disable</button>
                        </td>
                    </tr>
                </tbody>
            </table>
        `);
    }

    showPatients() {
        this.updateContent('Patient Management', 'Home > Patient Care > Patients', `
            <div class="info-panel">
                <h3><i class="fas fa-user-injured"></i> Patient Overview</h3>
                <p>Manage patient records and medical information.</p>
            </div>
            
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Patient ID</th>
                        <th>Name</th>
                        <th>Age</th>
                        <th>Condition</th>
                        <th>Last Visit</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>PHI-001</td>
                        <td>*** Protected ***</td>
                        <td>45</td>
                        <td>Knee Replacement</td>
                        <td>2024-01-15</td>
                        <td>
                            <button style="padding: 4px 8px; border: 1px solid var(--border-color); background: var(--white); border-radius: 4px; margin-right: 5px;">View</button>
                            <button style="padding: 4px 8px; border: 1px solid var(--border-color); background: var(--white); border-radius: 4px;">Edit</button>
                        </td>
                    </tr>
                </tbody>
            </table>
            
            <div class="warning-panel">
                <h3><i class="fas fa-shield-alt"></i> PHI Protection</h3>
                <p>Patient names and sensitive information are protected. Full access requires proper authentication.</p>
            </div>
        `);
    }

    showDocuments() {
        this.updateContent('Medical Documents', 'Home > Documents', `
            <div class="info-panel">
                <h3><i class="fas fa-file-medical"></i> Document Management</h3>
                <p>Access and manage medical documentation.</p>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 20px;">
                <div style="border: 1px solid var(--border-color); border-radius: 8px; padding: 15px;">
                    <h4><i class="fas fa-notes-medical"></i> Medical Notes</h4>
                    <p style="color: var(--medium-gray); font-size: 0.9rem;">Patient consultation notes and observations</p>
                    <button style="margin-top: 10px; padding: 6px 12px; background: var(--primary-color); color: white; border: none; border-radius: 4px;">Access</button>
                </div>
                
                <div style="border: 1px solid var(--border-color); border-radius: 8px; padding: 15px;">
                    <h4><i class="fas fa-x-ray"></i> Imaging Reports</h4>
                    <p style="color: var(--medium-gray); font-size: 0.9rem;">X-rays, MRI, and other imaging studies</p>
                    <button style="margin-top: 10px; padding: 6px 12px; background: var(--primary-color); color: white; border: none; border-radius: 4px;">Access</button>
                </div>
                
                <div style="border: 1px solid var(--border-color); border-radius: 8px; padding: 15px;">
                    <h4><i class="fas fa-file-prescription"></i> Lab Results</h4>
                    <p style="color: var(--medium-gray); font-size: 0.9rem;">Laboratory test results and analysis</p>
                    <button style="margin-top: 10px; padding: 6px 12px; background: var(--primary-color); color: white; border: none; border-radius: 4px;">Access</button>
                </div>
            </div>
        `);
    }

    showQuery() {
        this.updateContent('Medical Queries', 'Home > Tools > Query', `
            <div class="info-panel">
                <h3><i class="fas fa-search"></i> Medical Information Query</h3>
                <p>Search medical literature and patient data with AI assistance.</p>
            </div>
            
            <div style="margin-top: 20px;">
                <label style="display: block; margin-bottom: 10px; font-weight: 500;">Enter your medical query:</label>
                <textarea style="width: 100%; height: 100px; padding: 10px; border: 2px solid var(--border-color); border-radius: 6px; resize: vertical;" placeholder="e.g., Latest treatment options for ACL tears in athletes..."></textarea>
                <button style="margin-top: 10px; padding: 10px 20px; background: var(--primary-color); color: white; border: none; border-radius: 4px; cursor: pointer;">
                    <i class="fas fa-robot"></i> Query AI Assistant
                </button>
            </div>
            
            <div class="success-panel" style="margin-top: 20px;">
                <h3><i class="fas fa-info-circle"></i> AI-Powered Medical Search</h3>
                <p>This system will integrate with medical literature databases and patient records to provide comprehensive, evidence-based responses to clinical queries.</p>
            </div>
        `);
    }

    showAnalytics() {
        this.updateContent('Analytics Dashboard', 'Home > Analytics', `
            <div class="info-panel">
                <h3><i class="fas fa-chart-bar"></i> Medical Analytics</h3>
                <p>Data insights and reporting for medical operations.</p>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
                <div style="background: var(--light-gray); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="color: var(--primary-color); margin-bottom: 5px;">247</h3>
                    <p style="color: var(--medium-gray);">Total Patients</p>
                </div>
                <div style="background: var(--light-gray); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="color: var(--medical-green); margin-bottom: 5px;">94%</h3>
                    <p style="color: var(--medium-gray);">Treatment Success</p>
                </div>
                <div style="background: var(--light-gray); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="color: var(--accent-color); margin-bottom: 5px;">12</h3>
                    <p style="color: var(--medium-gray);">Active Cases</p>
                </div>
                <div style="background: var(--light-gray); padding: 20px; border-radius: 8px; text-align: center;">
                    <h3 style="color: var(--warning-color); margin-bottom: 5px;">3</h3>
                    <p style="color: var(--medium-gray);">Pending Reviews</p>
                </div>
            </div>
        `);
    }

    showSystem() {
        this.updateContent('System Settings', 'Home > Administration > System', `
            <div class="info-panel">
                <h3><i class="fas fa-cogs"></i> System Configuration</h3>
                <p>Manage system settings and configurations.</p>
            </div>
            
            <div style="margin-top: 20px;">
                <h4>Security Settings</h4>
                <div style="margin: 10px 0; padding: 10px; border: 1px solid var(--border-color); border-radius: 4px;">
                    <label style="display: flex; align-items: center;">
                        <input type="checkbox" checked style="margin-right: 10px;">
                        Enable MFA for PHI access
                    </label>
                </div>
                <div style="margin: 10px 0; padding: 10px; border: 1px solid var(--border-color); border-radius: 4px;">
                    <label style="display: flex; align-items: center;">
                        <input type="checkbox" checked style="margin-right: 10px;">
                        Audit all PHI access
                    </label>
                </div>
                
                <h4 style="margin-top: 20px;">Session Management</h4>
                <div style="margin: 10px 0; padding: 10px; border: 1px solid var(--border-color); border-radius: 4px;">
                    <label>Session timeout (minutes):</label>
                    <input type="number" value="15" style="margin-left: 10px; padding: 5px; border: 1px solid var(--border-color); border-radius: 4px;">
                </div>
            </div>
        `);
    }

    showAudit() {
        this.updateContent('Audit Logs', 'Home > Administration > Audit', `
            <div class="info-panel">
                <h3><i class="fas fa-clipboard-list"></i> Audit Trail</h3>
                <p>Review system access and security events.</p>
            </div>
            
            <table class="data-table" style="margin-top: 20px;">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>User</th>
                        <th>Action</th>
                        <th>Resource</th>
                        <th>IP Address</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>2024-01-15 14:30:25</td>
                        <td>dr_smith</td>
                        <td>PHI_READ</td>
                        <td>Patient-001</td>
                        <td>192.168.1.100</td>
                        <td><span style="color: var(--medical-green);">Success</span></td>
                    </tr>
                    <tr>
                        <td>2024-01-15 14:28:15</td>
                        <td>nurse_jones</td>
                        <td>USER_LOGIN</td>
                        <td>System</td>
                        <td>192.168.1.105</td>
                        <td><span style="color: var(--medical-green);">Success</span></td>
                    </tr>
                    <tr>
                        <td>2024-01-15 14:25:10</td>
                        <td>unknown</td>
                        <td>LOGIN_ATTEMPT</td>
                        <td>System</td>
                        <td>10.0.0.50</td>
                        <td><span style="color: var(--danger-color);">Failed</span></td>
                    </tr>
                </tbody>
            </table>
        `);
    }

    showWelcome() {
        this.updateContent('Welcome', 'Home', `
            <div class="welcome-panel">
                <h2>Welcome to Orthopedics EMR</h2>
                <p>Select an option from the menu to get started.</p>
            </div>
        `);
    }

    updateContent(title, breadcrumb, content) {
        document.getElementById('contentTitle').textContent = title;
        document.getElementById('breadcrumb').innerHTML = breadcrumb.split(' > ').map((item, index, arr) => 
            index === arr.length - 1 ? `<span style="color: var(--primary-color);">${item}</span>` : item
        ).join(' > ');
        document.getElementById('contentArea').innerHTML = content;
    }

    // Toast Messages
    showError(message) {
        this.showToast('errorToast', 'errorMessage', message);
    }

    showSuccess(message) {
        this.showToast('successToast', 'successMessage', message);
    }

    showToast(toastId, messageId, message) {
        const toast = document.getElementById(toastId);
        const messageElement = document.getElementById(messageId);
        
        messageElement.textContent = message;
        toast.classList.add('show');
        
        setTimeout(() => {
            toast.classList.remove('show');
        }, 5000);
    }

    hideToast(toast) {
        toast.classList.remove('show');
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new MedicalEMR();
});